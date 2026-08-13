"""BM25 召回：文档 = statement×3 + detail。

对疑问句会再跑一路「去停用词」查询，与原问做 RRF 融合，减轻
when/what/did 等高频疑问词冲淡内容词的问题（Velora / LoCoMo 同路径）。
"""

from __future__ import annotations

import math
from collections import Counter

from ..models import Atom
from .base import RecallHit, RecallOutcome
from .tokenize import tokenize

_K1 = 1.5
_B = 0.75
_RRF_K = 60
_OVERFETCH = 4

# 疑问/虚词：保留在原问一路，改写路丢掉以突出实体与事件词
_QUERY_STOP = frozenset(
    {
        "when",
        "what",
        "where",
        "who",
        "whom",
        "whose",
        "which",
        "how",
        "why",
        "did",
        "does",
        "do",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "the",
        "a",
        "an",
        "to",
        "in",
        "on",
        "of",
        "for",
        "and",
        "or",
        "but",
        "with",
        "from",
        "as",
        "at",
        "by",
        "into",
        "about",
        "her",
        "his",
        "she",
        "he",
        "they",
        "them",
        "their",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "would",
        "could",
        "should",
        "still",
        "also",
        "just",
        "have",
        "has",
        "had",
        "will",
        "can",
        "may",
        "might",
        "than",
        "then",
        "too",
        "very",
        "more",
        "most",
        "such",
        "own",
        "other",
        "some",
        "any",
        "all",
        "each",
        "few",
        "many",
        "much",
        "if",
        "so",
        "not",
        "no",
        "nor",
        "only",
        "same",
        "both",
        "between",
        "through",
        "during",
        "before",
        "after",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "now",
        "want",
        "pursue",
        "likely",
        "considered",
        "member",
        "community",
        "什么",
        "怎么",
        "怎样",
        "哪里",
        "哪儿",
        "哪个",
        "哪些",
        "谁",
        "何时",
        "为什么",
        "是否",
        "吗",
        "呢",
        "的",
        "了",
        "在",
        "是",
        "和",
        "与",
        "或",
        "被",
        "把",
        "对",
        "从",
        "到",
        "为",
        "也",
        "都",
        "就",
        "还",
        "很",
        "最",
        "更",
    }
)


def _doc_tokens(atom: Atom) -> list[str]:
    return tokenize(atom.statement) * 3 + tokenize(atom.detail or "")


def content_query(query: str) -> str:
    """去掉疑问/虚词后的内容向查询；若被掏空则回退原问。"""
    keep = [
        t
        for t in tokenize(query)
        if t not in _QUERY_STOP and len(t) > 2
    ]
    return " ".join(keep) if keep else query


def _rrf_fuse(
    ranked_lists: list[list[RecallHit]],
    *,
    k: int = _RRF_K,
    limit: int,
) -> list[RecallHit]:
    scores: dict[str, float] = {}
    best: dict[str, RecallHit] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            key = hit.atom.key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            prev = best.get(key)
            if prev is None or (hit.score or 0.0) > (prev.score or 0.0):
                best[key] = hit
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: list[RecallHit] = []
    for key, fused in ordered[:limit]:
        hit = best[key]
        out.append(RecallHit(atom=hit.atom, score=round(fused, 6)))
    return out


def _term_coverage(atom: Atom, terms: list[str]) -> int:
    if not terms:
        return 0
    blob_terms = set(_doc_tokens(atom))
    return sum(1 for t in terms if t in blob_terms)


class Bm25Recall:
    def retrieve(self, atoms: list[Atom], query: str, max_atoms: int) -> RecallOutcome:
        if not query.strip() or not atoms:
            return RecallOutcome()

        overfetch = max(max_atoms * _OVERFETCH, 20)
        primary = self._score(atoms, query, overfetch)
        rewritten = content_query(query)
        if rewritten.strip().lower() == query.strip().lower():
            candidates = primary
        else:
            secondary = self._score(atoms, rewritten, overfetch)
            if not secondary:
                candidates = primary
            else:
                # 过取融合，再按内容词覆盖重排截断
                candidates = _rrf_fuse(
                    [primary, secondary], limit=overfetch
                )

        focus = tokenize(rewritten)
        if focus:
            candidates = sorted(
                candidates,
                key=lambda h: (
                    _term_coverage(h.atom, focus),
                    h.score or 0.0,
                ),
                reverse=True,
            )
        return RecallOutcome(hits=candidates[:max_atoms])

    def _score(
        self, atoms: list[Atom], query: str, limit: int
    ) -> list[RecallHit]:
        terms = tokenize(query)
        if not terms:
            return []

        docs = [Counter(_doc_tokens(a)) for a in atoms]
        doc_lens = [sum(c.values()) for c in docs]
        avg_len = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0
        n = len(docs)

        df = Counter()
        for term in set(terms):
            df[term] = sum(1 for d in docs if term in d)

        scored: list[RecallHit] = []
        for atom, doc, dl in zip(atoms, docs, doc_lens):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * (_K1 + 1) / (
                    tf + _K1 * (1 - _B + _B * dl / avg_len)
                )
            if score > 0:
                scored.append(RecallHit(atom=atom, score=round(score, 4)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:limit]
