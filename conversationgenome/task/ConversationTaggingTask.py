from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import bittensor as bt
from pydantic import BaseModel

from conversationgenome.api.models.conversation import Conversation
from conversationgenome.llm.llm_factory import get_llm_backend
from conversationgenome.task.Task import Task


class ConversationTaskInputData(BaseModel):
    window_idx: int = -1
    window: Optional[List[Tuple[int, str]]] = None
    participants: Optional[List[str]] = None
    enrichment_lines: Optional[List[Tuple[int, str]]] = None


class ConversationTaskInput(BaseModel):
    guid: str
    input_type: Literal["conversation"]
    data: ConversationTaskInputData
    input_categories: Optional[List[str]] = None


class ConversationTaggingTask(Task):
    type: Literal["conversation_tagging"] = "conversation_tagging"
    input: Optional[ConversationTaskInput] = None

    async def mine(self) -> dict[str, list]:
        llml = get_llm_backend()

        try:
            conversation = Conversation(
                guid=self.input.guid,
                lines=self.input.data.window,
                miner_task_prompt=self.prompt_chain[0].prompt_template,
                input_categories=self.input.input_categories
            )

            result = llml.conversation_to_metadata(conversation=conversation, generateEmbeddings=False)

            enrichment_lines = self.input.data.enrichment_lines or []
            if not enrichment_lines:
                return {"tags": result.tags, "vectors": result.vectors}

            all_tags = []
            if result and result.tags:
                all_tags.append(result.tags)

            for _, content in enrichment_lines:
                enrichment_result = llml.enrichment_to_metadata(
                    content,
                    generateEmbeddings=False,
                    input_categories=self.input.input_categories,
                )
                if enrichment_result and enrichment_result.tags:
                    all_tags.append(enrichment_result.tags)

            if not all_tags:
                return {"tags": result.tags, "vectors": result.vectors}

            combined_result = llml.combine_metadata_tags(all_tags, generateEmbeddings=False)
            if combined_result:
                output = {"tags": combined_result.tags, "vectors": combined_result.vectors}
            else:
                output = {"tags": result.tags, "vectors": result.vectors}
        except Exception as e:
            bt.logging.error(f"Error during mining: {e}")
            raise e

        return output
