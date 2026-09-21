"""Document to DocumentTree. No model involved unless the file declares no structure.

Each parser recovers the hierarchy the file already carries. That is the honest version
of the method: the tree is the author's own outline, not one a model invented.

All parsers reduce a document to an ordered list of `Heading`s, each with a level, a
title, its own text (excluding descendants) and optional page and character spans. One
assembler turns that list into a tree, so nesting rules are identical across formats.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from navjev.types import DocumentTree, NodeId, TreeNode

StructureSource = Literal["native", "llm_inferred"]


class NoStructureError(ValueError):
    """The file carries no usable hierarchy and LLM inference was not allowed."""


@dataclass
class Heading:
    level: int
    title: str
    text: str
    page_span: tuple[int, int] | None = None
    char_span: tuple[int, int] | None = None


def assemble_tree(
    doc_id: str,
    title: str,
    preamble: str,
    headings: Sequence[Heading],
    source_path: str,
    parser: str,
    structure_source: StructureSource,
    preamble_page_span: tuple[int, int] | None = None,
) -> DocumentTree:
    """Nest headings by level. A heading's parent is the nearest preceding heading with a
    smaller level, so skipped levels (H1 to H3) still nest instead of breaking the tree.

    A document with exactly one level-1 heading that comes first is rooted at it, since
    that heading is the document title; otherwise the root is a synthetic node named after
    the document and every heading hangs beneath it.
    """
    nodes: dict[NodeId, TreeNode] = {}
    root = TreeNode("root", title, 0, preamble, page_span=preamble_page_span)
    nodes[root.id] = root

    items = list(headings)
    if items:
        top = min(h.level for h in items)
        first_is_sole_top = items[0].level == top and sum(h.level == top for h in items) == 1
        if first_is_sole_top:
            first = items.pop(0)
            root.title = first.title
            root.text = (
                (preamble + "\n\n" + first.text).strip() if preamble.strip() else first.text
            )
            root.page_span = _merge_spans(preamble_page_span, first.page_span)
            root.char_span = first.char_span

    # Stack of (level, node) from the root down.
    stack: list[tuple[int, TreeNode]] = [(0, root)]
    for i, h in enumerate(items, start=1):
        while len(stack) > 1 and stack[-1][0] >= h.level:
            stack.pop()
        parent = stack[-1][1]
        node = TreeNode(
            id=f"n{i:04d}",
            title=h.title.strip() or "(untitled)",
            depth=parent.depth + 1,
            text=h.text,
            page_span=h.page_span,
            char_span=h.char_span,
            parent_id=parent.id,
        )
        parent.children.append(node.id)
        nodes[node.id] = node
        stack.append((h.level, node))

    return DocumentTree(
        doc_id=doc_id,
        title=root.title,
        root_id=root.id,
        nodes=nodes,
        source_path=source_path,
        parser=parser,
        structure_source=structure_source,
    )


def _merge_spans(
    a: tuple[int, int] | None, b: tuple[int, int] | None
) -> tuple[int, int] | None:
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]))


class Parser(Protocol):
    extensions: tuple[str, ...]

    def parse(self, path: Path) -> DocumentTree: ...


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------

_ATX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.*?)(?:[ \t]+#+)?[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def markdown_headings(text: str) -> list[tuple[int, int, str]]:
    """(line index, level, title) for each ATX heading outside fenced code blocks."""
    out: list[tuple[int, int, str]] = []
    fence: str | None = None
    for i, line in enumerate(text.split("\n")):
        m = _FENCE.match(line)
        if m:
            marker = m.group(1)
            if fence is None:
                fence = marker
                continue
            if marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
                continue
        if fence is not None:
            continue
        h = _ATX.match(line)
        if h:
            out.append((i, len(h.group(1)), h.group(2).strip()))
    return out


class MarkdownParser:
    """ATX headings define the hierarchy. Fenced code blocks are not headings even when a
    line inside them starts with #, which is the bug every naive implementation ships."""

    extensions = (".md", ".markdown")

    def parse(self, path: Path) -> DocumentTree:
        text = path.read_text(encoding="utf-8")
        return self.parse_text(text, doc_id=path.stem, title=path.stem, source_path=str(path))

    def parse_text(
        self, text: str, doc_id: str, title: str, source_path: str = ""
    ) -> DocumentTree:
        lines = text.split("\n")
        # Character offset of the start of every line, for char spans.
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line) + 1)
        found = markdown_headings(text)
        if not found:
            raise NoStructureError(f"{source_path or doc_id}: no ATX headings found")

        preamble = "\n".join(lines[: found[0][0]]).strip()
        headings: list[Heading] = []
        for k, (line_idx, level, heading_title) in enumerate(found):
            end_line = found[k + 1][0] if k + 1 < len(found) else len(lines)
            body = "\n".join(lines[line_idx + 1 : end_line]).strip()
            start_char = offsets[line_idx]
            end_char = offsets[end_line] if end_line < len(offsets) else len(text)
            headings.append(Heading(level, heading_title, body, None, (start_char, end_char)))
        return assemble_tree(
            doc_id, title, preamble, headings, source_path, "markdown", "native"
        )


# --------------------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------------------


class PdfParser:
    """Uses the outline (bookmark) tree when present, and records page spans so a
    retrieved node can be shown to a person as pages.

    A PDF with no outline goes to the LLM fallback and is tagged `llm_inferred`, which
    the evaluation reports separately rather than mixing in.

    Text is cut at page granularity: a node owns the pages from its outline entry up to
    the page before the next entry. Two entries on one page share that page's text, which
    is a known coarseness of outline-based cutting and is recorded, not hidden.
    """

    extensions = (".pdf",)

    def __init__(self, structure_fallback: LlmStructureParser | None = None) -> None:
        self.structure_fallback = structure_fallback

    def parse(self, path: Path) -> DocumentTree:
        return _run(self.parse_async(path))

    async def parse_async(self, path: Path) -> DocumentTree:
        import fitz

        with fitz.open(path) as doc:
            pages = [page.get_text("text") for page in doc]
            toc = [(int(lvl), str(t), int(p)) for lvl, t, p in doc.get_toc(simple=True)]
            meta_title = (doc.metadata or {}).get("title") or ""
        title = meta_title.strip() or path.stem
        toc = [(lvl, t, p) for lvl, t, p in toc if 1 <= p <= len(pages)]
        if not toc:
            if self.structure_fallback is None:
                raise NoStructureError(
                    f"{path}: PDF has no outline; enable allow_llm_inferred_structure to "
                    f"infer one with an LLM"
                )
            return await self.structure_fallback.parse_pages_async(
                pages, doc_id=path.stem, title=title, source_path=str(path), parser="pdf"
            )

        first_page = toc[0][2]
        preamble = "\n".join(pages[: first_page - 1]).strip()
        preamble_span = (1, first_page - 1) if first_page > 1 else None
        headings: list[Heading] = []
        for k, (level, heading_title, page) in enumerate(toc):
            next_page = toc[k + 1][2] if k + 1 < len(toc) else len(pages) + 1
            last = max(page, next_page - 1)
            body = "\n".join(pages[page - 1 : last]).strip()
            headings.append(Heading(level, heading_title, body, (page, last)))
        return assemble_tree(
            path.stem, title, preamble, headings, str(path), "pdf", "native", preamble_span
        )


# --------------------------------------------------------------------------------------
# DOCX
# --------------------------------------------------------------------------------------

_DOCX_HEADING = re.compile(r"^Heading (\d)$", re.IGNORECASE)


class DocxParser:
    """Heading styles (`Heading 1` … `Heading 9`, plus `Title`) define the hierarchy."""

    extensions = (".docx",)

    def parse(self, path: Path) -> DocumentTree:
        import docx
        import docx.table

        document = docx.Document(str(path))
        blocks: list[tuple[int | None, str]] = []  # (heading level or None, text)
        for item in document.iter_inner_content():
            if isinstance(item, docx.table.Table):
                rows = []
                for row in item.rows:
                    rows.append("\t".join(cell.text.strip() for cell in row.cells))
                blocks.append((None, "\n".join(rows)))
                continue
            style = (item.style.name if item.style is not None else "") or ""
            m = _DOCX_HEADING.match(style)
            if style.lower() == "title":
                blocks.append((1, item.text))
            elif m:
                blocks.append((int(m.group(1)), item.text))
            else:
                blocks.append((None, item.text))

        found = [i for i, (lvl, _) in enumerate(blocks) if lvl is not None]
        if not found:
            raise NoStructureError(f"{path}: DOCX has no heading styles")
        preamble = "\n".join(t for _, t in blocks[: found[0]]).strip()
        headings: list[Heading] = []
        for k, idx in enumerate(found):
            end = found[k + 1] if k + 1 < len(found) else len(blocks)
            level, heading_title = blocks[idx]
            assert level is not None
            body = "\n".join(t for _, t in blocks[idx + 1 : end]).strip()
            headings.append(Heading(level, heading_title, body))
        core_title = (document.core_properties.title or "").strip() or path.stem
        return assemble_tree(
            path.stem, core_title, preamble, headings, str(path), "docx", "native"
        )


# --------------------------------------------------------------------------------------
# LLM-inferred structure
# --------------------------------------------------------------------------------------

STRUCTURE_PROMPT_VERSION = "v1"

STRUCTURE_SYSTEM = (
    "You recover the outline of a document from its plain text. You are given numbered "
    "lines. Return the section headings that appear in the text, in order, each with the "
    "1-based number of the line where the heading sits and a nesting level from 1 (top) "
    "to 4. Only report headings that literally appear in the text; never invent one. "
    "Prefer a heading per major section over many tiny ones."
)

STRUCTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line": {"type": "integer"},
                    "level": {"type": "integer"},
                    "title": {"type": "string"},
                },
                "required": ["line", "level", "title"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["headings"],
    "additionalProperties": False,
}


class LlmStructureParser:
    """Last resort for flat documents. Asks an LLM for a heading structure over the text.

    Sets `structure_source="llm_inferred"` on the tree. Never silently used: the CLI says
    so, and the manifest records how many documents in a run needed it.

    Long texts are sent in windows; the last few headings of the previous window are
    passed along so levels stay consistent across the seam.
    """

    extensions = (".txt",)

    def __init__(self, llm: Any, model: str, window_chars: int = 60_000) -> None:
        self.llm = llm
        self.model = model
        self.window_chars = window_chars
        self.usage_input_tokens = 0
        self.usage_output_tokens = 0
        self.cost_usd = 0.0

    def parse(self, path: Path) -> DocumentTree:
        return _run(self.parse_async(path))

    async def parse_async(self, path: Path) -> DocumentTree:
        text = path.read_text(encoding="utf-8", errors="replace")
        return await self.parse_pages_async(
            [text], doc_id=path.stem, title=path.stem, source_path=str(path)
        )

    async def parse_pages_async(
        self,
        pages: Sequence[str],
        doc_id: str,
        title: str,
        source_path: str = "",
        parser: str = "text",
    ) -> DocumentTree:
        lines: list[str] = []
        line_page: list[int] = []  # 1-based page per line
        for p, page in enumerate(pages, start=1):
            for line in page.split("\n"):
                lines.append(line)
                line_page.append(p)
        found = await self._infer_headings(lines)
        if not found:
            raise NoStructureError(f"{source_path or doc_id}: the LLM found no headings")

        preamble = "\n".join(lines[: found[0][0]]).strip()
        preamble_span = (line_page[0], line_page[found[0][0] - 1]) if found[0][0] > 0 else None
        headings: list[Heading] = []
        for k, (line_idx, level, heading_title) in enumerate(found):
            end_line = found[k + 1][0] if k + 1 < len(found) else len(lines)
            body = "\n".join(lines[line_idx + 1 : end_line]).strip()
            span = (line_page[line_idx], line_page[max(line_idx, end_line - 1)])
            headings.append(
                Heading(level, heading_title, body, span if len(pages) > 1 else None)
            )
        return assemble_tree(
            doc_id,
            title,
            preamble,
            headings,
            source_path,
            f"{parser}+llm_structure",
            "llm_inferred",
            preamble_span if len(pages) > 1 else None,
        )

    async def _infer_headings(self, lines: list[str]) -> list[tuple[int, int, str]]:
        found: list[tuple[int, int, str]] = []
        start = 0
        while start < len(lines):
            # Grow the window line by line until it reaches window_chars.
            size = 0
            end = start
            while end < len(lines) and size + len(lines[end]) + 1 <= self.window_chars:
                size += len(lines[end]) + 1
                end += 1
            if end == start:
                end = start + 1  # one pathological line longer than the window
            numbered = "\n".join(f"{i + 1}: {lines[i]}" for i in range(start, end))
            context = ""
            if found:
                tail = found[-5:]
                context = "Headings already found before this window:\n" + "\n".join(
                    f"- level {lvl}: {t}" for _, lvl, t in tail
                )
            user = f"{context}\n\nLines {start + 1} to {end}:\n{numbered}".strip()
            completion = await self.llm.complete_json(
                self.model, STRUCTURE_SYSTEM, user, STRUCTURE_SCHEMA, max_tokens=4096
            )
            self.usage_input_tokens += completion.input_tokens
            self.usage_output_tokens += completion.output_tokens
            self.cost_usd += completion.cost_usd
            for h in completion.data.get("headings", []):
                line = int(h["line"]) - 1
                if start <= line < end and (not found or line > found[-1][0]):
                    level = min(max(int(h["level"]), 1), 6)
                    found.append((line, level, str(h["title"]).strip()))
            start = end
        return found


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------


def _run(coro: Coroutine[Any, Any, DocumentTree]) -> DocumentTree:
    """Run a parser coroutine from synchronous code. Inside a running loop, callers must
    use the async entry points instead; this raises rather than deadlocking."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError("use parse_async() from inside an event loop")


async def parse_async(
    path: Path, structure_fallback: LlmStructureParser | None = None
) -> DocumentTree:
    """Dispatch on extension. `structure_fallback` is only consulted by parsers whose
    native structure can be absent, and only when the caller allowed it."""
    suffix = path.suffix.lower()
    if suffix in MarkdownParser.extensions:
        return MarkdownParser().parse(path)
    if suffix in PdfParser.extensions:
        return await PdfParser(structure_fallback).parse_async(path)
    if suffix in DocxParser.extensions:
        return DocxParser().parse(path)
    if suffix in LlmStructureParser.extensions:
        if structure_fallback is None:
            raise NoStructureError(
                f"{path}: plain text has no structure; enable allow_llm_inferred_structure"
            )
        return await structure_fallback.parse_async(path)
    raise ValueError(f"no parser for {suffix!r} ({path})")


def parse(path: Path, structure_fallback: LlmStructureParser | None = None) -> DocumentTree:
    """Synchronous `parse_async`."""
    return _run(parse_async(path, structure_fallback))
