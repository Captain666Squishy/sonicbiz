#!/usr/bin/env python3
"""Helper agent for managing Hugo blog content."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


KEY_PRIORITY = [
    "title",
    "summary",
    "description",
    "slug",
    "date",
    "lastmod",
    "draft",
    "tags",
    "categories",
    "layout",
    "image",
    "alt",
    "readingTime",
    "wordCount",
]

MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdx"}
FRONT_MATTER_DELIM = "+++"
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SCHEME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")


@dataclass
class MarkdownDocument:
    front_matter: Dict[str, object]
    body: str
    has_front_matter: bool


@dataclass
class LinkRef:
    target: str
    line: int


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_-]+", "-", value)
    return value.strip("-") or "untitled"


def load_markdown(path: Path) -> MarkdownDocument:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONT_MATTER_DELIM:
        return MarkdownDocument({}, text.strip("\n") + "\n", False)

    closing_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == FRONT_MATTER_DELIM:
            closing_idx = idx
            break
    if closing_idx is None:
        return MarkdownDocument({}, text.strip("\n") + "\n", False)

    fm_text = "\n".join(lines[1:closing_idx]).strip()
    body = "\n".join(lines[closing_idx + 1 :]).lstrip("\n")
    data = parse_front_matter_block(fm_text) if fm_text else {}
    return MarkdownDocument(dict(data), body, True)


def parse_front_matter_block(text: str) -> Dict[str, object]:
    data: Dict[str, object] = {}
    lines = text.splitlines()
    idx = 0
    pending_key: Optional[str] = None
    pending_value: List[str] = []

    while idx < len(lines):
        raw_line = lines[idx]
        idx += 1
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if pending_key is None:
            if "=" not in raw_line:
                continue
            key, value = raw_line.split("=", 1)
            pending_key = key.strip()
            pending_value = [value.strip()]
        else:
            pending_value.append(raw_line)

        combined = "\n".join(pending_value)
        if needs_more_lines(combined):
            continue

        data[pending_key] = coerce_value(combined)
        pending_key = None
        pending_value = []

    if pending_key is not None:
        data[pending_key] = coerce_value("\n".join(pending_value))

    return data


def needs_more_lines(value: str) -> bool:
    stripped = value.rstrip()
    if not stripped:
        return False

    triple_double = stripped.count('"""')
    triple_single = stripped.count("'''")
    if triple_double % 2 == 1 or triple_single % 2 == 1:
        return True

    bracket_balance = brace_balance = 0
    in_string = False
    string_char = ""
    escape = False
    for char in value:
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if in_string:
            if char == string_char:
                in_string = False
            continue
        if char in ("'", '"'):
            in_string = True
            string_char = char
            continue
        if char == "[":
            bracket_balance += 1
        elif char == "]":
            bracket_balance = max(bracket_balance - 1, 0)
        elif char == "{":
            brace_balance += 1
        elif char == "}":
            brace_balance = max(brace_balance - 1, 0)

    if bracket_balance > 0 or brace_balance > 0:
        return True

    return stripped.endswith(",")


def parse_string_literal(value: str) -> str:
    try:
        return ast.literal_eval(value)
    except Exception:
        return value.strip('"').strip("'")


def split_top_level(text: str) -> List[str]:
    items: List[str] = []
    current: List[str] = []
    depth = 0
    in_string = False
    string_char = ""
    escape = False
    for char in text:
        if escape:
            current.append(char)
            escape = False
            continue
        if char == "\\" and in_string:
            current.append(char)
            escape = True
            continue
        if in_string:
            current.append(char)
            if char == string_char:
                in_string = False
            continue
        if char in ("'", '"'):
            in_string = True
            string_char = char
            current.append(char)
            continue
        if char in "[{(":
            depth += 1
            current.append(char)
            continue
        if char in "]})":
            depth = max(depth - 1, 0)
            current.append(char)
            continue
        if char == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def parse_inline_table(value: str) -> Dict[str, object]:
    inner = value.strip()[1:-1].strip()
    if not inner:
        return {}
    items = split_top_level(inner)
    table: Dict[str, object] = {}
    for item in items:
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        table[key.strip()] = coerce_value(val.strip())
    return table


def parse_array(value: str) -> List[object]:
    inner = value.strip()[1:-1].strip()
    if not inner:
        return []
    items = split_top_level(inner)
    return [coerce_value(item) for item in items]


def coerce_value(value: str) -> object:
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped.startswith(("'''", '"""')):
        if stripped.count(stripped[:3]) >= 2:
            return parse_string_literal(stripped)
    if stripped[0] in {"'", '"'}:
        return parse_string_literal(stripped)
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if stripped.startswith("[") and stripped.endswith("]"):
        return parse_array(stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        return parse_inline_table(stripped)
    if re.fullmatch(r"[-+]?\d+", stripped):
        try:
            return int(stripped)
        except ValueError:
            pass
    if re.fullmatch(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", stripped) or re.fullmatch(
        r"[-+]?\d+(?:[eE][-+]?\d+)", stripped
    ):
        try:
            return float(stripped)
        except ValueError:
            pass
    return stripped


def toml_escape(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\t", "\\t")
    escaped = escaped.replace("\n", "\\n")
    return f'"{escaped}"'


def format_toml_value(value: object) -> str:
    if isinstance(value, str):
        return toml_escape(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, dt.time):
        return value.isoformat()
    if isinstance(value, Path):
        return toml_escape(str(value))
    if isinstance(value, dict):
        inner = ", ".join(f"{k} = {format_toml_value(v)}" for k, v in value.items())
        return "{ " + inner + " }"
    if isinstance(value, list):
        inner = ", ".join(format_toml_value(v) for v in value)
        return "[ " + inner + " ]"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def ordered_items(data: Dict[str, object]) -> List[Tuple[str, object]]:
    def sort_key(item: Tuple[str, object]) -> Tuple[int, str]:
        key = item[0]
        try:
            priority = KEY_PRIORITY.index(key)
        except ValueError:
            priority = len(KEY_PRIORITY)
        return (priority, key)

    return sorted(data.items(), key=sort_key)


def dumps_front_matter(data: Dict[str, object]) -> str:
    lines = []
    for key, value in ordered_items(data):
        lines.append(f"{key} = {format_toml_value(value)}")
    return "\n".join(lines)


def write_markdown(path: Path, data: Dict[str, object], body: str) -> None:
    fm = dumps_front_matter(data)
    body_str = body.rstrip() + "\n"
    content = f"{FRONT_MATTER_DELIM}\n{fm}\n{FRONT_MATTER_DELIM}\n\n{body_str}"
    path.write_text(content, encoding="utf-8")


def iter_markdown_files(target: Path) -> Iterable[Path]:
    if target.is_file() and target.suffix.lower() in MARKDOWN_EXTENSIONS:
        yield target
        return
    if target.is_dir():
        for path in target.rglob("*"):
            if path.is_file() and path.suffix.lower() in MARKDOWN_EXTENSIONS:
                yield path


def clean_markdown_for_summary(body: str) -> str:
    text = re.sub(r"```.*?```", "", body, flags=re.S)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"[#>*_-]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_paragraph(body: str, limit: int = 200) -> str:
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        plain = clean_markdown_for_summary(block)
        if not plain:
            continue
        if len(plain) > limit:
            return plain[: limit - 1].rstrip() + "…"
        return plain
    return ""


def count_words(body: str) -> int:
    cleaned = clean_markdown_for_summary(body)
    return len(re.findall(r"\b[\w'-]+\b", cleaned))


def format_markdown_text(text: str) -> Tuple[str, bool]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    result: List[str] = []
    idx = 0
    total = len(lines)

    while idx < total:
        line = lines[idx]
        stripped = line.lstrip()
        if stripped.startswith("#"):
            if result and result[-1] != "":
                result.append("")
            result.append(line)
            next_line = lines[idx + 1] if idx + 1 < total else ""
            if next_line.strip() and not next_line.lstrip().startswith("#"):
                result.append("")
        else:
            if stripped.startswith("* "):
                line = re.sub(r"^\s*\*", "-", line)
            result.append(line)
        idx += 1

    cleaned: List[str] = []
    blank = False
    for line in result:
        if line.strip():
            cleaned.append(line)
            blank = False
        else:
            if not blank:
                cleaned.append("")
            blank = True

    formatted = "\n".join(cleaned).strip("\n") + "\n"
    changed = formatted != text
    return formatted, changed


def extract_markdown_links(text: str) -> List[LinkRef]:
    refs: List[LinkRef] = []
    for match in LINK_PATTERN.finditer(text):
        raw_target = match.group(1).strip()
        line = text.count("\n", 0, match.start()) + 1
        refs.append(LinkRef(target=raw_target, line=line))
    return refs


def normalize_link_target(raw: str) -> str:
    target = raw.strip()
    if not target:
        return ""
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    for sep in ('" ', "' "):
        idx = target.find(sep)
        if idx != -1:
            target = target[:idx].strip()
            break
    target = target.split("#", 1)[0]
    target = target.split("?", 1)[0]
    return target.strip()


def is_internal_link(target: str) -> bool:
    if not target:
        return False
    lower = target.lower()
    if target.startswith("#"):
        return False
    if target.startswith("{{"):
        return False
    if SCHEME_PATTERN.match(target):
        return False
    if lower.startswith(("mailto:", "tel:", "javascript:", "data:")):
        return False
    return True


class BlogAgent:
    def __init__(self, repo_root: Path) -> None:
        self.root = repo_root
        self.content_dir = self.root / "content"
        self.layouts_dir = self.root / "layouts"
        self.themes_dir = self.root / "themes"
        self.static_dir = self.root / "static"
        self.assets_dir = self.root / "assets"
        self.resources_dir = self.root / "resources"

    # --- new post ---------------------------------------------------------
    def create_post(self, args: argparse.Namespace) -> Path:
        slug = args.slug or slugify(args.title)
        section = args.section.strip("/").rstrip("/") or "blog"
        post_dir = self.content_dir / section / slug
        post_dir.mkdir(parents=True, exist_ok=True)
        index_path = post_dir / "index.md"

        if index_path.exists() and not args.force:
            raise SystemExit(f"{index_path} already exists (use --force to overwrite).")

        now = dt.datetime.now(dt.timezone.utc)
        front_matter: Dict[str, object] = {
            "title": args.title,
            "slug": slug,
            "date": now.isoformat(),
            "draft": not args.publish,
        }
        if args.summary:
            front_matter["summary"] = args.summary.strip()
            front_matter.setdefault("description", args.summary.strip())
        if args.description:
            front_matter["description"] = args.description.strip()
        if args.tags:
            front_matter["tags"] = [tag.strip() for tag in args.tags]
        if args.categories:
            front_matter["categories"] = [cat.strip() for cat in args.categories]
        if args.layout:
            front_matter["layout"] = args.layout
        if args.image:
            front_matter["image"] = args.image
        if args.alt_text:
            front_matter["alt"] = args.alt_text

        body = ""
        if args.body_file:
            body = Path(args.body_file).read_text(encoding="utf-8")
        elif args.body:
            body = args.body
        else:
            prompt = args.summary or "Share something interesting."
            body = f"## {args.title}\n\n{prompt}\n"

        write_markdown(index_path, front_matter, body)

        if args.auto_metadata:
            self.update_metadata(argparse.Namespace(
                target=index_path,
                summary_len=args.summary_len,
                words_per_minute=args.words_per_minute,
                dry_run=False,
                fill_summary=True,
                set_lastmod=True,
            ))

        return index_path

    # --- metadata --------------------------------------------------------
    def update_metadata(self, args: argparse.Namespace) -> None:
        target = Path(args.target)
        files = list(iter_markdown_files(target))
        if not files:
            raise SystemExit(f"No markdown files found in {target}.")

        now = dt.datetime.now(dt.timezone.utc)
        for path in files:
            doc = load_markdown(path)
            front = dict(doc.front_matter)
            body = doc.body
            changed = False

            slug = front.get("slug")
            if not isinstance(slug, str) or not slug:
                inferred = (
                    path.parent.name if path.name == "index.md" else slugify(path.stem)
                )
                front["slug"] = inferred
                changed = True

            if args.fill_summary and not front.get("summary"):
                summary = first_paragraph(body, limit=args.summary_len)
                if summary:
                    front["summary"] = summary
                    front.setdefault("description", summary)
                    changed = True

            if args.set_lastmod:
                front["lastmod"] = now.isoformat()
                changed = True

            if "date" not in front:
                stat = path.stat()
                created = dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc)
                front["date"] = created.isoformat()
                changed = True

            words = count_words(body)
            reading = max(1, math.ceil(words / args.words_per_minute))
            if front.get("wordCount") != words:
                front["wordCount"] = words
                changed = True
            if front.get("readingTime") != reading:
                front["readingTime"] = reading
                changed = True

            if not changed:
                continue

            if args.dry_run:
                print(f"[dry-run] Would update metadata in {path}")
            else:
                write_markdown(path, front, body)
                print(f"Updated metadata in {path}")

    # --- markdown formatting ---------------------------------------------
    def format_markdown(self, args: argparse.Namespace) -> None:
        target = Path(args.target)
        files = list(iter_markdown_files(target))
        if not files:
            raise SystemExit(f"No markdown files found in {target}.")

        dirty = []
        for path in files:
            original = path.read_text(encoding="utf-8")
            formatted, changed = format_markdown_text(original)
            if not changed:
                continue
            if args.check:
                dirty.append(path)
                continue
            path.write_text(formatted, encoding="utf-8")
            print(f"Formatted {path}")

        if args.check and dirty:
            joined = "\n".join(str(p) for p in dirty)
            raise SystemExit(f"Markdown changes needed:\n{joined}")

    # --- layout suggestions ----------------------------------------------
    def suggest_layouts(self, args: argparse.Namespace) -> None:
        section = args.section
        layouts = self._discover_layouts(section)
        if not layouts:
            print(f"No layouts referencing '{section}' were found.")
            return
        widest = max(len(name) for name, *_ in layouts)
        print(f"Suggested layouts for '{section}':\n")
        for name, source, description in layouts:
            print(f"- {name.ljust(widest)}  ({source})  {description}")

    def _discover_layouts(self, section: str) -> List[Tuple[str, str, str]]:
        candidates: List[Tuple[str, str, str]] = []
        search_roots: List[Tuple[Path, str]] = []
        if self.layouts_dir.exists():
            search_roots.append((self.layouts_dir, "layouts"))
        if self.themes_dir.exists():
            for theme in sorted(self.themes_dir.iterdir()):
                layout_dir = theme / "layouts"
                if layout_dir.exists():
                    search_roots.append((layout_dir, f"themes/{theme.name}"))

        for root, label in search_roots:
            for file in root.rglob("*.html"):
                rel = file.relative_to(root)
                parts = rel.parts
                if section not in parts and "_default" not in parts and "partials" not in parts:
                    continue
                description = describe_layout(rel)
                name = "/".join(parts)
                candidates.append((name, label, description))
        return candidates

    # --- audits ----------------------------------------------------------
    def audit(self, args: argparse.Namespace) -> None:
        target = Path(args.target)
        files = list(iter_markdown_files(target))
        if not files:
            raise SystemExit(f"No markdown files found in {target}.")

        missing_front_matter: List[Path] = []
        broken_links: List[Tuple[Path, int, str]] = []

        for path in files:
            doc = load_markdown(path)
            if not doc.has_front_matter:
                missing_front_matter.append(path)
            broken = self._find_broken_links(path, doc.body)
            for line, link in broken:
                broken_links.append((path, line, link))

        if missing_front_matter:
            print("Missing front matter detected:")
            for path in missing_front_matter:
                print(f"- {path}")
            print("")
        else:
            print("All markdown files contain front matter.\n")

        if broken_links:
            print("Broken internal links:")
            for file_path, line, link in broken_links:
                print(f"- {file_path}:{line} -> {link}")
        else:
            print("No broken internal links found.")

        if missing_front_matter or broken_links:
            raise SystemExit(1)

    def _find_broken_links(self, path: Path, body: str) -> List[Tuple[int, str]]:
        issues: List[Tuple[int, str]] = []
        for ref in extract_markdown_links(body):
            normalized = normalize_link_target(ref.target)
            if not is_internal_link(normalized):
                continue
            if not normalized:
                continue
            if not self._link_target_exists(normalized, path):
                issues.append((ref.line, normalized))
        return issues

    def _link_target_exists(self, target: str, doc_path: Path) -> bool:
        clean = target.replace("%20", " ").strip()
        if not clean:
            return True

        if clean.startswith("/"):
            rel = clean.lstrip("/")
            return self._path_exists_from_roots(rel)

        doc_dir = doc_path.parent
        relative_candidate = (doc_dir / clean).resolve()
        if self._path_exists_absolute(relative_candidate):
            return True

        repo_candidate = (self.root / clean).resolve()
        if self._path_exists_absolute(repo_candidate):
            return True

        content_candidate = (self.content_dir / clean).resolve()
        if self._path_exists_absolute(content_candidate):
            return True

        return False

    def _path_exists_from_roots(self, relative: str) -> bool:
        rel = relative.strip("/")
        if not rel:
            return True
        candidates = [
            self.content_dir / rel,
            self.static_dir / rel,
            self.assets_dir / rel,
            self.resources_dir / rel,
        ]
        return any(self._path_exists_absolute(path) for path in candidates)

    @staticmethod
    def _path_exists_absolute(path: Path) -> bool:
        if path.exists():
            return True
        if path.suffix == "":
            if (path / "index.md").exists() or (path / "_index.md").exists():
                return True
            if path.with_suffix(".md").exists():
                return True
        return False


def describe_layout(rel_path: Path) -> str:
    text = rel_path.as_posix()
    if "single" in text:
        return "Single content view"
    if "list" in text:
        return "Section list view"
    if "card" in text:
        return "Card-based partial"
    if "summary" in text:
        return "Summary/preview component"
    if "baseof" in text:
        return "Base template"
    return "General template"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent for Hugo blog workflows.")
    parser.add_argument(
        "--repo",
        default=Path.cwd(),
        type=Path,
        help="Path to the Hugo repository (defaults to current working directory).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    new_parser = subparsers.add_parser("new", help="Generate a new post skeleton.")
    new_parser.add_argument("title", help="Human-friendly post title.")
    new_parser.add_argument("--section", default="blog", help="Content section (default: blog).")
    new_parser.add_argument("--slug", help="Custom slug (defaults to slugified title).")
    new_parser.add_argument("--summary", help="Optional summary/lede.")
    new_parser.add_argument("--description", help="Optional meta description (defaults to summary).")
    new_parser.add_argument("--tags", nargs="*", help="List of tags.")
    new_parser.add_argument("--categories", nargs="*", help="List of categories.")
    new_parser.add_argument("--layout", help="Preferred layout.")
    new_parser.add_argument("--image", help="Featured image path.")
    new_parser.add_argument("--alt-text", help="Alt text for featured image.")
    new_parser.add_argument("--body", help="Body text to seed the post.")
    new_parser.add_argument("--body-file", help="Path to a file used as the body.")
    new_parser.add_argument("--publish", action="store_true", help="Mark as published (draft = false).")
    new_parser.add_argument("--force", action="store_true", help="Overwrite existing post if present.")
    new_parser.add_argument(
        "--auto-metadata",
        action="store_true",
        help="Immediately run metadata automation on the new post.",
    )
    new_parser.add_argument(
        "--summary-len",
        type=int,
        default=200,
        help="Max length when auto-generating summaries.",
    )
    new_parser.add_argument(
        "--words-per-minute",
        type=int,
        default=200,
        help="Reading speed used for metadata calculations.",
    )

    format_parser = subparsers.add_parser(
        "format", help="Fix minor Markdown formatting issues."
    )
    format_parser.add_argument(
        "target",
        help="File or directory to format.",
    )
    format_parser.add_argument(
        "--check",
        action="store_true",
        help="Only report files that would change.",
    )

    layout_parser = subparsers.add_parser(
        "layouts", help="Suggest available layouts/partials for a section."
    )
    layout_parser.add_argument(
        "--section",
        default="blog",
        help="Section to inspect (default: blog).",
    )

    metadata_parser = subparsers.add_parser(
        "metadata", help="Automate metadata for markdown files."
    )
    metadata_parser.add_argument(
        "target",
        help="File or directory to update.",
    )
    metadata_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing files.",
    )
    metadata_parser.add_argument(
        "--fill-summary",
        action="store_true",
        help="Populate summary/description if missing.",
    )
    metadata_parser.add_argument(
        "--summary-len",
        type=int,
        default=200,
        help="Max summary length when filling.",
    )
    metadata_parser.add_argument(
        "--words-per-minute",
        type=int,
        default=200,
        help="Reading time speed.",
    )
    metadata_parser.add_argument(
        "--set-lastmod",
        action="store_true",
        help="Always refresh the lastmod timestamp.",
    )

    audit_parser = subparsers.add_parser(
        "audit", help="Check markdown for missing front matter and broken internal links."
    )
    audit_parser.add_argument(
        "target",
        nargs="?",
        default="content",
        help="File or directory to audit (default: content).",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo.expanduser().resolve()
    if not repo_root.exists():
        raise SystemExit(f"Repository path {repo_root} does not exist.")

    agent = BlogAgent(repo_root)

    if args.command == "new":
        created = agent.create_post(args)
        print(f"Created {created.relative_to(repo_root)}")
    elif args.command == "format":
        agent.format_markdown(args)
    elif args.command == "layouts":
        agent.suggest_layouts(args)
    elif args.command == "metadata":
        agent.update_metadata(args)
    elif args.command == "audit":
        agent.audit(args)
    else:  # pragma: no cover
        parser.error("Unknown command.")


if __name__ == "__main__":
    main()
