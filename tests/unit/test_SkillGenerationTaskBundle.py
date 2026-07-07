from unittest.mock import AsyncMock, Mock, patch
import json
import pytest
from conversationgenome.task_bundle.SkillGenerationTaskBundle import SkillGenerationTaskBundle
from tests.mocks.DummyData import DummyData


def test_is_ready_false_when_no_metadata():
    bundle = DummyData.skill_generation_task_bundle()
    bundle.input.metadata = None
    assert not bundle.is_ready()


def test_is_ready_false_when_no_indexed_windows():
    bundle = DummyData.skill_generation_task_bundle()
    bundle.input.data.indexed_windows = None
    assert not bundle.is_ready()


def test_is_ready_true_when_metadata_and_indexed_windows():
    bundle = DummyData.setup_skill_generation_task_bundle()
    assert bundle.is_ready()


@pytest.mark.asyncio
async def test_setup_calls_trim_input_and_split_and_generate():
    bundle = DummyData.skill_generation_task_bundle()
    with patch.object(bundle, '_split_conversation_in_windows') as mock_split, \
         patch.object(bundle, '_enforce_minimum_convo_windows') as mock_enforce, \
         patch.object(bundle, '_generate_metadata') as mock_generate:

        await bundle.setup()

        mock_split.assert_called_once()
        mock_enforce.assert_called_once()
        mock_generate.assert_called_once()


@pytest.mark.asyncio
async def test_setup_user_request_skips_min_windows():
    bundle = DummyData.skill_generation_task_bundle()
    bundle.is_user_request = True
    with patch.object(bundle, '_split_conversation_in_windows') as mock_split, \
         patch.object(bundle, '_enforce_minimum_convo_windows') as mock_enforce, \
         patch.object(bundle, '_generate_metadata') as mock_generate:

        await bundle.setup()

        mock_split.assert_called_once()
        mock_enforce.assert_not_called()
        mock_generate.assert_called_once()


def test_to_mining_tasks_creates_tasks():
    bundle = DummyData.setup_skill_generation_task_bundle()
    tasks = bundle.to_mining_tasks(2)
    assert len(tasks) == 2
    for task in tasks:
        assert task.type == "skill_generation"
        assert task.bundle_guid == bundle.guid
        assert task.input.input_type == "skill"


def test_generate_result_logs_counts_tags_and_vectors():
    bundle = DummyData.setup_skill_generation_task_bundle()
    miner_result = {"tags": ["tag1", "tag2"], "vectors": {"tag1": [0.1]}, "original_tags": ["tag1", "tag2", "tag3"]}
    log = bundle.generate_result_logs(miner_result)
    assert "tags: 2" in log
    assert "vector count: 1" in log
    assert "original tags: 3" in log


@pytest.mark.asyncio
async def test_format_results_validates_and_embeds_tags():
    bundle = DummyData.setup_skill_generation_task_bundle()
    miner_result = {"tags": ["tag1", "tag2"]}
    with patch('conversationgenome.task_bundle.SkillGenerationTaskBundle.get_llm_backend') as mock_llm_factory:
        mock_llm = Mock()
        mock_llm.validate_tag_set.return_value = ["tag1", "tag2"]
        mock_llm_factory.return_value = mock_llm
        with patch.object(bundle, '_get_vector_embeddings_set', AsyncMock(return_value={"tag1": [0.1], "tag2": [0.2]})):
            result = await bundle.format_results(miner_result)
    assert result["original_tags"] == ["tag1", "tag2"]
    assert result["tags"] == ["tag1", "tag2"]
    assert result["vectors"] == {"tag1": [0.1], "tag2": [0.2]}


@pytest.mark.asyncio
async def test_evaluate_calls_ground_truth_scoring():
    bundle = DummyData.setup_skill_generation_task_bundle()
    with patch('conversationgenome.task_bundle.SkillGenerationTaskBundle.GroundTruthTagSimilarityScoringMechanism') as mock_mech:
        mock_eval = AsyncMock(return_value="score")
        mock_mech.return_value.evaluate = mock_eval
        result = await bundle.evaluate(["response"])
    assert result == "score"


def test_enforce_minimum_convo_windows_sets_empty_when_below_minimum():
    bundle = DummyData.setup_skill_generation_task_bundle()
    bundle.input.data.min_convo_windows = 3
    bundle.input.data.indexed_windows = [(0, []), (1, [])]  # Only 2 windows
    bundle._enforce_minimum_convo_windows()
    assert bundle.input.data.indexed_windows == []


@pytest.mark.asyncio
async def test_generate_metadata_calls_llm_and_sets_metadata():
    bundle = DummyData.skill_generation_task_bundle()
    bundle.input.data.lines = [(0, json.dumps({
        "seed": "Skill to parse docx",
        "skill_markdown": "# Parse docx\n\nInstructions.",
        "enrichment": {},
    }))]

    with patch('conversationgenome.task_bundle.SkillGenerationTaskBundle.get_llm_backend') as mock_llm_factory:
        mock_llm = Mock()

        skill_result = Mock()
        skill_result.tags = ["docx", "parsing"]
        mock_llm.skill_to_metadata.return_value = skill_result

        combine_result = Mock()
        combine_result.success = True
        combine_result.tags = ["docx", "parsing"]
        combine_result.vectors = {"docx": {"vectors": [0.1]}}
        mock_llm.combine_metadata_tags.return_value = combine_result

        mock_llm_factory.return_value = mock_llm

        await bundle._generate_metadata()

        assert bundle.input.metadata.tags == ["docx", "parsing"]
        assert bundle.input.metadata.vectors == {"docx": {"vectors": [0.1]}}

        # skill_markdown is tagged; seed is ignored
        mock_llm.skill_to_metadata.assert_called_once_with("# Parse docx\n\nInstructions.", input_categories=None)
        mock_llm.combine_metadata_tags.assert_called_once_with([["docx", "parsing"]], generateEmbeddings=True)

        # input lines rewritten to the plain skill markdown for the miner window
        assert bundle.input.data.lines == [(0, "# Parse docx\n\nInstructions.")]


@pytest.mark.asyncio
async def test_generate_metadata_handles_failure():
    bundle = DummyData.skill_generation_task_bundle()
    bundle.input.data.lines = [(0, json.dumps({"seed": "s", "skill_markdown": "md", "enrichment": {}}))]
    bundle.input.metadata = None
    with patch('conversationgenome.task_bundle.SkillGenerationTaskBundle.get_llm_backend') as mock_llm_factory:
        mock_llm = Mock()
        skill_result = Mock()
        skill_result.tags = ["md"]
        mock_llm.skill_to_metadata.return_value = skill_result

        combine_result = Mock()
        combine_result.success = False
        mock_llm.combine_metadata_tags.return_value = combine_result

        mock_llm_factory.return_value = mock_llm

        await bundle._generate_metadata()

        assert bundle.input.metadata is None


@pytest.mark.asyncio
async def test_generate_metadata_handles_no_result():
    bundle = DummyData.skill_generation_task_bundle()
    bundle.input.data.lines = [(0, json.dumps({"seed": "s", "skill_markdown": "md", "enrichment": {}}))]
    bundle.input.metadata = None
    with patch('conversationgenome.task_bundle.SkillGenerationTaskBundle.get_llm_backend') as mock_llm_factory:
        mock_llm = Mock()
        skill_result = Mock()
        skill_result.tags = ["md"]
        mock_llm.skill_to_metadata.return_value = skill_result
        mock_llm.combine_metadata_tags.return_value = None

        mock_llm_factory.return_value = mock_llm

        await bundle._generate_metadata()

        assert bundle.input.metadata is None


@pytest.mark.asyncio
async def test_get_vector_embeddings_set_calls_llm():
    bundle = DummyData.setup_skill_generation_task_bundle()
    mock_llm = Mock()
    mock_llm.get_vector_embeddings_set.return_value = {"tag": [0.1]}
    result = await bundle._get_vector_embeddings_set(llml=mock_llm, tags=["tag"])
    assert result == {"tag": [0.1]}
