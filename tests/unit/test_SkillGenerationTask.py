from unittest.mock import MagicMock, Mock
from unittest.mock import patch

import pytest

from conversationgenome.prompt_chain.PromptChainStep import PromptChainStep
from conversationgenome.task.SkillGenerationTask import SkillGenerationTask, SkillTaskInput, SkillTaskInputData


def _make_task(window):
    return SkillGenerationTask(
        mode="local",
        api_version=1.4,
        guid="test-guid",
        bundle_guid="bundle-guid",
        type="skill_generation",
        input=SkillTaskInput(
            guid="input-guid",
            input_type="skill",
            data=SkillTaskInputData(
                window=window,
                participants=[]
            )
        ),
        prompt_chain=[PromptChainStep(
            step=0,
            id="skill_001",
            crc=12345,
            title="Infer tags",
            name="infer_tags_for_skill",
            description="Infer descriptive tags for a skill document",
            type="inference",
            input_path="skill",
            prompt_template="Infer tags for the skill",
            output_variable="final_output",
            output_type="List[str]"
        )]
    )


@pytest.mark.asyncio
async def test_mine_returns_tags_and_vectors():
    task = _make_task([(0, "# Parse .docx\n\nInstructions to parse docx files.")])

    mock_llml = MagicMock()
    skill_result = Mock()
    skill_result.tags = ["docx", "parsing"]
    mock_llml.skill_to_metadata = Mock(return_value=skill_result)

    combined_result = Mock()
    combined_result.tags = ["docx", "parsing"]
    combined_result.vectors = {"docx": [0.1], "parsing": [0.2]}
    mock_llml.combine_metadata_tags = Mock(return_value=combined_result)

    with patch("conversationgenome.task.SkillGenerationTask.get_llm_backend", return_value=mock_llml):
        result = await task.mine()

    assert result["tags"] == ["docx", "parsing"]
    assert result["vectors"] == {"docx": [0.1], "parsing": [0.2]}
    mock_llml.skill_to_metadata.assert_called_once_with(
        "# Parse .docx\n\nInstructions to parse docx files.", generateEmbeddings=False, input_categories=None
    )
    mock_llml.combine_metadata_tags.assert_called_once_with([["docx", "parsing"]], generateEmbeddings=False)


@pytest.mark.asyncio
async def test_mine_handles_empty_window():
    task = _make_task([])
    mock_llml = MagicMock()

    with patch("conversationgenome.task.SkillGenerationTask.get_llm_backend", return_value=mock_llml):
        result = await task.mine()

    assert result["tags"] == []
    assert result["vectors"] is None
    mock_llml.skill_to_metadata.assert_not_called()
    mock_llml.combine_metadata_tags.assert_not_called()


@pytest.mark.asyncio
async def test_mine_handles_none_result():
    task = _make_task([(0, "Test skill content")])
    mock_llml = MagicMock()
    mock_llml.skill_to_metadata = Mock(return_value=None)
    mock_llml.combine_metadata_tags = Mock(return_value=None)

    with patch("conversationgenome.task.SkillGenerationTask.get_llm_backend", return_value=mock_llml):
        result = await task.mine()

    assert result["tags"] == []
    assert result["vectors"] is None


@pytest.mark.asyncio
async def test_mine_raises_on_llm_exception():
    task = _make_task([(0, "Test skill content")])
    mock_llml = MagicMock()
    mock_llml.skill_to_metadata = Mock(side_effect=Exception("LLM Error"))

    with patch("conversationgenome.task.SkillGenerationTask.get_llm_backend", return_value=mock_llml):
        with pytest.raises(Exception, match="LLM Error"):
            await task.mine()
