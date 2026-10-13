"""LLM 召回：读索引点名 key。"""

from ..consolidation import prompts
from pydantic import BaseModel
from ..consolidation.engine import parse_json_object, _json_schema_format
from ..llm.base import ChatLLM, LLMError
from ..models import Atom
from .base import RecallHit, RecallOutcome


class RecallError(Exception):
    pass


class RecallPlan(BaseModel):
    keys: list[str]


class LlmRecall:
    def __init__(self, llm: ChatLLM):
        self._llm = llm

    def retrieve(self, atoms: list[Atom], query: str, max_atoms: int) -> RecallOutcome:
        if not atoms:
            return RecallOutcome()
        try:
            index = "\n".join(f"- [{atom.kind.value}] {atom.key}: {atom.statement}\n  detail: {atom.detail or ''}" for atom in atoms)
            res = self._llm.complete(
                prompts.RECALL_SYSTEM.format(max_atoms=max_atoms),
                f"## atom 索引与详情\n{index}\n\n## 当前情境\n{query}",
                response_format=_json_schema_format("recall_plan", RecallPlan),
            )
            data = parse_json_object(res.text)
            keys = list(dict.fromkeys(RecallPlan.model_validate(data).keys))[:max_atoms]
        except (LLMError, ValueError) as e:
            raise RecallError(str(e)) from e
        by_key = {a.key: a for a in atoms}
        if any(key not in by_key for key in keys):
            raise RecallError("Model selected an unknown memory key")
        hits = [RecallHit(atom=by_key[k]) for k in dict.fromkeys(keys)]
        return RecallOutcome(
            hits=hits,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
        )
