"""LLM 召回：读索引点名 key。"""

from ..consolidation import prompts
from ..consolidation.engine import parse_json_object
from ..llm.base import ChatLLM, LLMError
from ..models import Atom
from .base import RecallHit, RecallOutcome


class RecallError(Exception):
    pass


class LlmRecall:
    def __init__(self, llm: ChatLLM):
        self._llm = llm

    def retrieve(self, atoms: list[Atom], query: str, max_atoms: int) -> RecallOutcome:
        if not atoms:
            return RecallOutcome()
        try:
            res = self._llm.complete(
                prompts.RECALL_SYSTEM.format(max_atoms=max_atoms),
                f"## atom 索引\n{prompts.render_index(atoms)}\n\n## 当前情境\n{query}",
            )
            data = parse_json_object(res.text)
            keys = (data.get("keys") or [])[:max_atoms]
        except (LLMError, ValueError) as e:
            raise RecallError(str(e)) from e
        by_key = {a.key: a for a in atoms}
        hits = [RecallHit(atom=by_key[k]) for k in keys if k in by_key]
        return RecallOutcome(
            hits=hits,
            prompt_tokens=res.prompt_tokens,
            completion_tokens=res.completion_tokens,
        )
