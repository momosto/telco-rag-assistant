"""Load documents (Markdown, text, PDF, web pages) and split them into chunks.

Markdown is split on headings so each chunk is one section with a readable title,
which is what the citations show. Long sections are split again on paragraphs.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

MAX_CHARS = 1200


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str  # file name or URL
    title: str  # "Document title > Section"
    text: str


def _split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        if current and len(current) + len(para) + 2 > limit:
            parts.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_markdown(source: str, markdown: str) -> list[Chunk]:
    doc_title = source
    sections: list[tuple[str, list[str]]] = []
    heading, lines = None, []

    for line in markdown.splitlines():
        match = re.match(r"^(#{1,3})\s+(.*)", line)
        if match:
            level, text = len(match.group(1)), match.group(2).strip()
            if level == 1:
                doc_title = text
                continue
            if heading is not None or any(l.strip() for l in lines):
                sections.append((heading, lines))
            heading, lines = text, []
        else:
            lines.append(line)
    sections.append((heading, lines))

    chunks = []
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        title = f"{doc_title} > {heading}" if heading else doc_title
        for part in _split_long(body):
            chunks.append(Chunk(f"{source}#{len(chunks)}", source, title, part))
    return chunks


def chunk_plain(source: str, title: str, text: str) -> list[Chunk]:
    return [Chunk(f"{source}#{i}", source, title, part) for i, part in enumerate(_split_long(text.strip()))]


def load_pdf(path: Path) -> list[Chunk]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    chunks = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            for part in chunk_plain(f"{path.name} p{page_no}", f"{path.stem}, page {page_no}", text):
                chunks.append(part)
    return chunks


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "noscript"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "li", "h1", "h2", "h3", "tr", "br", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def chunk_html(url: str, page: str) -> list[Chunk]:
    parser = _TextExtractor()
    parser.feed(page)
    text = html.unescape(re.sub(r"\n\s*\n+", "\n\n", "".join(parser.parts)))
    return chunk_plain(url, parser.title.strip() or url, text)


def load_url(url: str) -> list[Chunk]:
    import httpx

    response = httpx.get(url, follow_redirects=True, timeout=20)
    response.raise_for_status()
    return chunk_html(url, response.text)


def load_directory(directory: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.iterdir()):
        if path.name.lower() == "readme.md":
            continue
        if path.suffix in {".md", ".txt"}:
            chunks.extend(chunk_markdown(path.name, path.read_text(encoding="utf-8")))
        elif path.suffix == ".pdf":
            chunks.extend(load_pdf(path))
    return chunks
