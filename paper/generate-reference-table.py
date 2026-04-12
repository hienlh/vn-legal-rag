#!/usr/bin/env python3
"""Parse references-formatted.tex and generate a verification table for reviewer."""

import re
from pathlib import Path

# Known URLs for each reference (for reviewer verification)
KNOWN_URLS = {
    "r1": "https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7f3d0b-Abstract.html",
    "r2": "https://aclanthology.org/D19-1410/",
    "r4": "https://dl.acm.org/doi/10.1145/3637528.3671470",
    "r5": "https://github.com/HKUDS/LightRAG",
    "r6": "https://arxiv.org/abs/2404.16130",
    "r7": "https://pageindex.ai/blog/pageindex-intro",
    "r8": "https://doi.org/10.5220/0011088400003176",
    "r9": "https://doi.org/10.1007/978-94-007-0120-5",
    "r10": "https://aclanthology.org/2020.emnlp-main.550/",
    "r11": "https://openreview.net/forum?id=hSyW5go0v8",
    "r13": "https://aclanthology.org/2023.acl-long.99/",
    "r16": "https://doi.org/10.1137/140976649",
    "r17": "https://doi.org/10.1145/775152.775191",
    "r18": "https://doi.org/10.1561/1500000019",
    "r19": "https://nlp.stanford.edu/IR-book/",
    "r20": "https://aclanthology.org/2020.findings-emnlp.261/",
    "r21": "https://doi.org/10.1142/S0218194018500304",
    "r22": "https://aclanthology.org/2020.findings-emnlp.92/",
    "r23": "https://aclanthology.org/N18-5012/",
    "r24": "https://doi.org/10.1109/KSE53942.2021.9648712",
    "r25": "https://aclanthology.org/N19-1423/",
    "r26": "https://aclanthology.org/2020.acl-main.466/",
    "r27": "https://doi.org/10.1016/j.aiopen.2024.09.002",
    "r28": "https://doi.org/10.1145/1571941.1572114",
}

NOTE_FLAGS = {
    "r5": "EMNLP 2025 (verified via GitHub)",
    "r6": "MS Research Technical Report (no peer-reviewed venue)",
    "r7": "Blog post only (no academic paper found)",
    "r27": "Updated: AI Open journal (was arXiv)",
}


def parse_references(tex_path: str) -> list[dict]:
    """Parse bibitem entries from thebibliography."""
    text = Path(tex_path).read_text(encoding="utf-8")
    entries = []
    # Split by \bibitem
    parts = re.split(r"\\bibitem\{(\w+)\}\s*\n?", text)
    # parts[0] is preamble, then alternating key/content
    for i in range(1, len(parts), 2):
        key = parts[i]
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        # Clean up LaTeX
        content = re.sub(r"\\end\{thebibliography\}", "", content).strip()
        # Extract author (before first period)
        author_match = re.match(r"^(.+?)\.\s", content)
        author = author_match.group(1) if author_match else "?"
        # Extract title (between first and second period, roughly)
        rest = content[len(author) + 2:] if author_match else content
        title_match = re.match(r"^(.+?)\.\s", rest)
        title = title_match.group(1) if title_match else rest.split(".")[0]
        entries.append({
            "key": key,
            "author": author,
            "title": title,
            "full": content,
            "url": KNOWN_URLS.get(key, ""),
            "note": NOTE_FLAGS.get(key, ""),
        })
    return entries


def generate_markdown_table(entries: list[dict]) -> str:
    """Generate markdown table for reviewer verification."""
    lines = [
        "# Reference Verification Table",
        "",
        "Generated from `references-formatted.tex` for reviewer hallucination check.",
        "",
        "| STT | Key | Author | Title | Link | Note |",
        "|-----|-----|--------|-------|------|------|",
    ]
    for i, e in enumerate(entries, 1):
        note = e["note"]
        url = e["url"]
        link = f"[link]({url})" if url else ""
        # Escape pipes in content
        author = e["author"].replace("|", "\\|")
        title = e["title"].replace("|", "\\|")
        lines.append(f"| {i} | {e['key']} | {author} | {title} | {link} | {note} |")
    return "\n".join(lines)


def generate_csv_table(entries: list[dict]) -> str:
    """Generate CSV for easy spreadsheet import."""
    lines = ["STT,Key,Author,Title,Link,Note"]
    for i, e in enumerate(entries, 1):
        # Escape quotes in CSV
        author = e["author"].replace('"', '""')
        title = e["title"].replace('"', '""')
        note = e["note"].replace('"', '""')
        lines.append(f'{i},{e["key"]},"{author}","{title}",{e["url"]},"{note}"')
    return "\n".join(lines)


if __name__ == "__main__":
    tex_path = Path(__file__).parent / "references-formatted.tex"
    entries = parse_references(str(tex_path))

    # Generate markdown
    md = generate_markdown_table(entries)
    md_path = tex_path.parent / "reference-verification-table.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"Markdown table: {md_path}")

    # Generate CSV
    csv_content = generate_csv_table(entries)
    csv_path = tex_path.parent / "reference-verification-table.csv"
    csv_path.write_text(csv_content, encoding="utf-8")
    print(f"CSV table: {csv_path}")

    # Print markdown to console
    print()
    print(md)
