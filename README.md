# sonicbiz
Sonic Business Ltd Hugo Website.

## Local workflow

Initialize the Blowfish theme submodule before running the site:

```bash
git submodule update --init --recursive
```

Run the local development server:

```bash
hugo server
```

Deploy by pushing to `main`. GitHub Actions builds the site and publishes GitHub Pages from the generated artifact, so `public/` and `resources/_gen/` are not source-of-truth files.

## Custom layouts

This repo intentionally overrides a small part of Blowfish:

- `layouts/index.html` for the branded homepage
- `layouts/_default/list.html` for the blog archive and category tabs
- `layouts/partials/article-link/card.html` for reusable article cards
- `assets/css/custom.css` for the site-specific visual layer

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
- Audit content: `python3 tools/blog_agent.py audit` (find missing front matter and broken internal links)

The agent works inside the repo root by default; use `--repo /path/to/site` to target another Hugo project.
