"""Core data structures.

These are deliberately plain: a tree of nodes, a decision record, and a trace. Nothing
here touches the network, and nothing here knows Jev exists.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

NodeId = str


@dataclass
class TreeNode:
    """One section of a document.

    `text` is the node's own content, excluding descendants. Traversal never reads it;
    only `title` and `summary` are sent to the decision model. It is fetched at the end,
    for the nodes that were selected, to build the answering context.
    """

    id: NodeId
    title: str
    depth: int
    text: str
    summary: str | None = None
    page_span: tuple[int, int] | None = None
    char_span: tuple[int, int] | None = None
    parent_id: NodeId | None = None
    children: list[NodeId] = field(default_factory=list)

    def content_hash(self) -> str:
        """SHA-256 of title plus text. Keys the summary cache."""
        h = hashlib.sha256()
        h.update(self.title.encode("utf-8"))
        h.update(b"\x00")
        h.update(self.text.encode("utf-8"))
        return h.hexdigest()

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class DocumentTree:
    doc_id: str
    title: str
    root_id: NodeId
    nodes: dict[NodeId, TreeNode]
    source_path: str
    parser: str
    structure_source: Literal["native", "llm_inferred"]
    """`llm_inferred` marks a document whose hierarchy an LLM guessed rather than one the
    file declared. Runs must not mix the two without saying so: the inference changes what
    the comparison measures."""

    @property
    def root(self) -> TreeNode:
        return self.nodes[self.root_id]

    def children_of(self, node_id: NodeId) -> list[TreeNode]:
        return [self.nodes[cid] for cid in self.nodes[node_id].children]

    def path_to(self, node_id: NodeId) -> list[str]:
        """Titles from the root down to this node, for the traversal state."""
        titles: list[str] = []
        current: NodeId | None = node_id
        while current is not None:
            node = self.nodes[current]
            titles.append(node.title)
            current = node.parent_id
        titles.reverse()
        return titles

    def ancestors_of(self, node_id: NodeId) -> list[TreeNode]:
        """Root first, excluding the node itself."""
        out: list[TreeNode] = []
        current = self.nodes[node_id].parent_id
        while current is not None:
            node = self.nodes[current]
            out.append(node)
            current = node.parent_id
        out.reverse()
        return out

    def is_descendant(self, node_id: NodeId, ancestor_id: NodeId) -> bool:
        current = self.nodes[node_id].parent_id
        while current is not None:
            if current == ancestor_id:
                return True
            current = self.nodes[current].parent_id
        return False

    def walk(self, node_id: NodeId | None = None) -> Iterator[TreeNode]:
        """Depth-first, pre-order, in document order."""
        start = self.root_id if node_id is None else node_id
        stack = [start]
        while stack:
            node = self.nodes[stack.pop()]
            yield node
            stack.extend(reversed(node.children))

    def leaves(self) -> list[TreeNode]:
        return [n for n in self.walk() if n.is_leaf]

    def max_depth(self) -> int:
        return max(n.depth for n in self.nodes.values())

    def to_dict(self) -> dict[str, Any]:
        nodes = {nid: asdict(n) for nid, n in self.nodes.items()}
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "root_id": self.root_id,
            "source_path": self.source_path,
            "parser": self.parser,
            "structure_source": self.structure_source,
            "nodes": nodes,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentTree:
        nodes: dict[NodeId, TreeNode] = {}
        for nid, raw in data["nodes"].items():
            page_span = raw.get("page_span")
            char_span = raw.get("char_span")
            nodes[nid] = TreeNode(
                id=raw["id"],
                title=raw["title"],
                depth=int(raw["depth"]),
                text=raw.get("text", ""),
                summary=raw.get("summary"),
                page_span=(int(page_span[0]), int(page_span[1])) if page_span else None,
                char_span=(int(char_span[0]), int(char_span[1])) if char_span else None,
                parent_id=raw.get("parent_id"),
                children=list(raw.get("children", [])),
            )
        structure_source = data["structure_source"]
        if structure_source not in ("native", "llm_inferred"):
            raise ValueError(f"unknown structure_source {structure_source!r}")
        return cls(
            doc_id=data["doc_id"],
            title=data["title"],
            root_id=data["root_id"],
            nodes=nodes,
            source_path=data["source_path"],
            parser=data["parser"],
            structure_source=structure_source,
        )

    @classmethod
    def from_json(cls, blob: str) -> DocumentTree:
        return cls.from_dict(json.loads(blob))


@dataclass
class ChildDecision:
    """One Score answer about one child, after normalization."""

    child_id: NodeId
    raw_score: float
    normalized: float
    """raw_score divided by the top level index, so thresholds compare across scales."""
    confidence: float
    probabilities: dict[str, float]
    expanded: bool


@dataclass
class Expansion:
    """Everything decided at one node, in one Jev request."""

    node_id: NodeId
    depth: int
    children: list[ChildDecision]
    stop_here: float
    """Noul probability that this node already answers the query."""
    escalated_to_llm: bool
    input_tokens: int
    latency_ms: float
    model_id: str
    """The versioned ID that actually answered, e.g. jev-1.13.0. Thresholds are only
    valid against the version they were fitted on, so it is recorded per expansion."""
    dead_end: bool = False
    off_topic: float | None = None
    """Noul probability that the query is unrelated to the document. Root only."""
    cost_usd: float = 0.0
    emitted: bool = False
    """Set by the beam when this node was emitted as a result."""
    jev_children: list[ChildDecision] | None = None
    jev_stop_here: float | None = None
    """When the expansion was escalated, the Jev answers it replaced. Kept so calibration
    can be measured on every Jev decision, including the ones the LLM overrode."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trace:
    query: str
    doc_id: str
    expansions: list[Expansion] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(e.input_tokens for e in self.expansions)

    @property
    def total_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.expansions)

    @property
    def escalation_rate(self) -> float:
        if not self.expansions:
            return 0.0
        return sum(1 for e in self.expansions if e.escalated_to_llm) / len(self.expansions)

    @property
    def model_ids(self) -> list[str]:
        seen: list[str] = []
        for e in self.expansions:
            if e.model_id not in seen:
                seen.append(e.model_id)
        return seen

    def as_path_text(self, tree: DocumentTree | None = None) -> str:
        """Human-readable retrieval path, for the CLI and for the paper's figure."""

        def title(node_id: NodeId) -> str:
            if tree is not None and node_id in tree.nodes:
                return tree.nodes[node_id].title
            return node_id

        lines = [f"query: {self.query}", f"doc: {self.doc_id}"]
        for e in self.expansions:
            indent = "  " * e.depth
            flags = []
            if e.escalated_to_llm:
                flags.append("escalated")
            if e.dead_end:
                flags.append("dead end")
            if e.emitted:
                flags.append("EMITTED")
            flag_text = f" [{', '.join(flags)}]" if flags else ""
            off = f" off_topic={e.off_topic:.2f}" if e.off_topic is not None else ""
            lines.append(
                f"{indent}{title(e.node_id)}  stop={e.stop_here:.2f}{off}"
                f"  model={e.model_id}{flag_text}"
            )
            for c in e.children:
                mark = "->" if c.expanded else "  "
                lines.append(
                    f"{indent}  {mark} {title(c.child_id)}  "
                    f"score={c.normalized:.2f} conf={c.confidence:.2f}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "doc_id": self.doc_id,
            "expansions": [e.to_dict() for e in self.expansions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        expansions = []
        for raw in data.get("expansions", []):
            children = [ChildDecision(**c) for c in raw.get("children", [])]
            jev_children = raw.get("jev_children")
            fields = {k: v for k, v in raw.items() if k not in ("children", "jev_children")}
            expansions.append(
                Expansion(
                    children=children,
                    jev_children=[ChildDecision(**c) for c in jev_children]
                    if jev_children is not None
                    else None,
                    **fields,
                )
            )
        return cls(query=data["query"], doc_id=data["doc_id"], expansions=expansions)


@dataclass
class RetrievalResult:
    nodes: list[TreeNode]
    trace: Trace
    latency_ms: float
    cost_usd: float
    no_answer: bool = False
    """True when the root-level off_topic test fired: the query is not about this
    document. An honest miss, which vector retrieval cannot express."""
    scores: list[float] = field(default_factory=list)
    """The score that admitted each node in `nodes`, parallel to it. For ranking only."""
    query_tokens: int = 0
    """Tokens spent by a non-Jev retriever on the query itself (dense arm embeddings)."""

    @property
    def node_ids(self) -> list[NodeId]:
        return [n.id for n in self.nodes]
