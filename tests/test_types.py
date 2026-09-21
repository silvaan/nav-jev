from __future__ import annotations

from navjev.types import ChildDecision, DocumentTree, Expansion, Trace, TreeNode


def make_tree() -> DocumentTree:
    nodes = {
        "root": TreeNode("root", "Doc", 0, "intro", children=["a", "b"]),
        "a": TreeNode("a", "A", 1, "text a", parent_id="root", children=["a1"]),
        "a1": TreeNode("a1", "A.1", 2, "text a1", parent_id="a", page_span=(3, 4)),
        "b": TreeNode("b", "B", 1, "text b", parent_id="root"),
    }
    return DocumentTree("doc", "Doc", "root", nodes, "doc.md", "markdown", "native")


def test_walk_is_preorder_document_order() -> None:
    assert [n.id for n in make_tree().walk()] == ["root", "a", "a1", "b"]


def test_path_and_ancestry() -> None:
    tree = make_tree()
    assert tree.path_to("a1") == ["Doc", "A", "A.1"]
    assert [n.id for n in tree.ancestors_of("a1")] == ["root", "a"]
    assert tree.is_descendant("a1", "root")
    assert not tree.is_descendant("b", "a")
    assert [n.id for n in tree.leaves()] == ["a1", "b"]


def test_json_roundtrip_preserves_spans() -> None:
    tree = make_tree()
    back = DocumentTree.from_json(tree.to_json())
    assert back.nodes["a1"].page_span == (3, 4)
    assert back.to_dict() == tree.to_dict()


def test_content_hash_depends_on_title_and_text() -> None:
    a = TreeNode("x", "T", 0, "body").content_hash()
    assert a == TreeNode("y", "T", 3, "body").content_hash()
    assert a != TreeNode("x", "T2", 0, "body").content_hash()
    assert a != TreeNode("x", "T", 0, "body2").content_hash()


def test_trace_aggregates_and_roundtrips() -> None:
    e1 = Expansion(
        "root",
        0,
        [ChildDecision("a", 2.7, 0.9, 0.8, {"3": 0.75}, True)],
        0.1,
        False,
        400,
        90.0,
        "jev-1.13.0",
        off_topic=0.02,
        cost_usd=0.00001,
    )
    e2 = Expansion("a", 1, [], 0.9, True, 300, 1200.0, "jev-1.13.0", emitted=True)
    trace = Trace("q", "doc", [e1, e2])
    assert trace.total_input_tokens == 700
    assert trace.escalation_rate == 0.5
    assert trace.model_ids == ["jev-1.13.0"]
    back = Trace.from_dict(trace.to_dict())
    assert back.to_dict() == trace.to_dict()
    text = trace.as_path_text(make_tree())
    assert "EMITTED" in text and "escalated" in text and "-> A" in text
