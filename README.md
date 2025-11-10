# sonicbiz
Sonic Business Ltd Hugo Website.

## Blog agent

Use `tools/blog_agent.py` to automate common Hugo blogging tasks (Python 3.9+).

### Quick start

```bash
python3 tools/blog_agent.py --help
```

### Key workflows

- Generate a post: `python3 tools/blog_agent.py new "Post Title" --summary "..." --tags tag1 tag2 --publish`
- Clean Markdown: `python3 tools/blog_agent.py format content/blog --check` (omit `--check` to write changes)
- Refresh metadata: `python3 tools/blog_agent.py metadata content/blog --fill-summary --set-lastmod`
- Explore layouts: `python3 tools/blog_agent.py layouts --section blog`

The agent works inside the repo root by default; use `--repo /path/to/site` to target another Hugo project.
