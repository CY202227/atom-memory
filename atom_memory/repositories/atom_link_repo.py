"""atom_link 仓储：类型化边的增删查与 derived_from 闭包。"""

from __future__ import annotations

from sqlmodel import Session, select

from ..models import Atom, AtomLink, AtomLinkKind, AtomStatus


def add(
    session: Session,
    *,
    space_id: int,
    src_atom_id: int,
    dst_atom_id: int,
    kind: AtomLinkKind | str,
    note: str | None = None,
) -> AtomLink | None:
    """幂等写入一条边；src==dst 或已存在则返回已有/None。"""
    if src_atom_id == dst_atom_id:
        return None
    kind_e = AtomLinkKind(kind) if isinstance(kind, str) else kind
    existing = session.exec(
        select(AtomLink).where(
            AtomLink.src_atom_id == src_atom_id,
            AtomLink.dst_atom_id == dst_atom_id,
            AtomLink.kind == kind_e,
        )
    ).first()
    if existing is not None:
        if note is not None and existing.note != note:
            existing.note = note
            session.add(existing)
        return existing
    row = AtomLink(
        space_id=space_id,
        src_atom_id=src_atom_id,
        dst_atom_id=dst_atom_id,
        kind=kind_e,
        note=note,
    )
    session.add(row)
    return row


def list_out(session: Session, atom_id: int) -> list[AtomLink]:
    return list(
        session.exec(
            select(AtomLink).where(AtomLink.src_atom_id == atom_id)
        ).all()
    )


def list_in(session: Session, atom_id: int) -> list[AtomLink]:
    return list(
        session.exec(
            select(AtomLink).where(AtomLink.dst_atom_id == atom_id)
        ).all()
    )


def neighbors(
    session: Session,
    atom_id: int,
    *,
    kinds: list[AtomLinkKind] | None = None,
) -> dict[str, list[dict]]:
    """按 kind 分组返回出边/入边（含对端 key）。"""
    out_rows = list_out(session, atom_id)
    in_rows = list_in(session, atom_id)
    if kinds is not None:
        kind_set = set(kinds)
        out_rows = [r for r in out_rows if r.kind in kind_set]
        in_rows = [r for r in in_rows if r.kind in kind_set]

    atom_ids = {r.dst_atom_id for r in out_rows} | {r.src_atom_id for r in in_rows}
    by_id: dict[int, Atom] = {}
    if atom_ids:
        for a in session.exec(select(Atom).where(Atom.id.in_(list(atom_ids)))).all():
            by_id[a.id] = a

    grouped: dict[str, list[dict]] = {
        "out": [],
        "in": [],
    }
    for r in out_rows:
        peer = by_id.get(r.dst_atom_id)
        grouped["out"].append(
            {
                "kind": r.kind.value,
                "atom_id": r.dst_atom_id,
                "key": peer.key if peer else None,
                "note": r.note,
            }
        )
    for r in in_rows:
        peer = by_id.get(r.src_atom_id)
        grouped["in"].append(
            {
                "kind": r.kind.value,
                "atom_id": r.src_atom_id,
                "key": peer.key if peer else None,
                "note": r.note,
            }
        )
    return grouped


def derived_atom_ids(session: Session, space_id: int) -> set[int]:
    """有至少一条 derived_from 出边的 atom id（派生 atom）。"""
    rows = session.exec(
        select(AtomLink.src_atom_id).where(
            AtomLink.space_id == space_id,
            AtomLink.kind == AtomLinkKind.derived_from,
        )
    ).all()
    return set(rows)


def reverse_derived_closure(
    session: Session,
    seed_atom_ids: set[int],
) -> set[int]:
    """沿 derived_from 反向闭包：谁 derived_from 了 seed（递归）。

    边方向：src derived_from dst → src 依赖 dst。
    反向：从 dst 找所有 src。
    """
    if not seed_atom_ids:
        return set()
    affected: set[int] = set()
    frontier = set(seed_atom_ids)
    while frontier:
        rows = session.exec(
            select(AtomLink.src_atom_id).where(
                AtomLink.dst_atom_id.in_(list(frontier)),
                AtomLink.kind == AtomLinkKind.derived_from,
            )
        ).all()
        nxt = {sid for sid in rows if sid not in affected and sid not in seed_atom_ids}
        # also exclude already in frontier iteration
        nxt -= affected
        if not nxt:
            break
        affected |= nxt
        frontier = nxt
    return affected


def neighbor_atoms(
    session: Session,
    atom_ids: list[int],
    *,
    kinds: list[AtomLinkKind] | None = None,
) -> list[Atom]:
    """命中集合一跳邻居（出边+入边），仅 active。"""
    if not atom_ids:
        return []
    kind_filter = kinds or [AtomLinkKind.about, AtomLinkKind.derived_from]
    peer_ids: set[int] = set()
    for aid in atom_ids:
        for r in list_out(session, aid):
            if r.kind in kind_filter:
                peer_ids.add(r.dst_atom_id)
        for r in list_in(session, aid):
            if r.kind in kind_filter:
                peer_ids.add(r.src_atom_id)
    peer_ids -= set(atom_ids)
    if not peer_ids:
        return []
    return list(
        session.exec(
            select(Atom).where(
                Atom.id.in_(list(peer_ids)),
                Atom.status == AtomStatus.active,
            )
        ).all()
    )


def delete_for_atom(session: Session, atom_id: int) -> None:
    rows = session.exec(
        select(AtomLink).where(
            (AtomLink.src_atom_id == atom_id) | (AtomLink.dst_atom_id == atom_id)
        )
    ).all()
    for row in rows:
        session.delete(row)


def replace_out_links(
    session: Session,
    *,
    space_id: int,
    src_atom_id: int,
    links: list[tuple[int, AtomLinkKind, str | None]],
) -> None:
    """用新出边集合替换该 atom 的全部出边（按本次写入）。"""
    existing = list_out(session, src_atom_id)
    for row in existing:
        session.delete(row)
    session.flush()
    for dst_id, kind, note in links:
        add(
            session,
            space_id=space_id,
            src_atom_id=src_atom_id,
            dst_atom_id=dst_id,
            kind=kind,
            note=note,
        )
