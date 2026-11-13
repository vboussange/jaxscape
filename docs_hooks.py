"""MkDocs hook that renders `README.md` as the landing page of the documentation.

`README.md` is the one source of the landing page: for `docs/index.md`, a
placeholder, this hook replaces the page's Markdown with the README's. It drops
each block between `<!-- docs:exclude-start -->` and `<!-- docs:exclude-end -->`,
and rewrites the README's links, outside code blocks, for the documentation:

- `docs/<path>` to `<path>`, for a documentation page or an image;
- any other repository path to its page on GitHub;
- `https://vboussange.github.io/jaxscape/<page>/` to the page's source file, so
  that the link works in a local preview and the strict build checks it.

A link that resolves to no file of the documentation logs a warning, which
fails the strict build.
"""

import logging
import re
from pathlib import Path, PurePosixPath


log = logging.getLogger("mkdocs.hooks.readme")

DOCS_URL = "https://vboussange.github.io/jaxscape/"
GITHUB_BLOB = "https://github.com/vboussange/jaxscape/blob/main/"
SITE_SUFFIXES = {".md", ".ipynb", ".png", ".gif", ".jpg", ".jpeg", ".svg"}

EXCLUDED = re.compile(
    r"^<!-- docs:exclude-start -->$.*?^<!-- docs:exclude-end -->$\n?", re.M | re.S
)
CODE_BLOCK = re.compile(r"(^```.*?^```$)", re.M | re.S)
# The target of a Markdown link or image, or of an HTML `src` or `href`.
LINK = re.compile(r"(?<=\]\()[^)\s]+(?=\))|(?<=src=\")[^\"]+|(?<=href=\")[^\"]+")


def _docs_page(path, files):
    """The source file of the published page `path`, or `None`."""
    path = path.strip("/")
    candidates = [f"{path}.md", f"{path}/index.md", f"{path}.ipynb"] if path else []
    for candidate in candidates or ["index.md"]:
        if files.get_file_from_path(candidate) is not None:
            return candidate
    return None


def _rewrite(url, files):
    path, hash_, anchor = url.partition("#")
    if url.startswith(DOCS_URL):
        target = _docs_page(path[len(DOCS_URL) :], files)
    elif re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:|#|/", url):
        return url  # An external URL, an anchor of this page, or an absolute path.
    elif path.startswith("docs/") and PurePosixPath(path).suffix in SITE_SUFFIXES:
        target = path.removeprefix("docs/")
        target = target if files.get_file_from_path(target) is not None else None
    else:
        return GITHUB_BLOB + url
    if target is None:
        log.warning("README.md links to '%s', which is no file of the docs.", url)
        return url
    return target + hash_ + anchor


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri != "index.md":
        return markdown
    readme = Path(config.config_file_path).parent / "README.md"
    text = EXCLUDED.sub("", readme.read_text(encoding="utf-8"))
    parts = CODE_BLOCK.split(text)
    # Odd parts are code blocks, which keep their text.
    for i in range(0, len(parts), 2):
        parts[i] = LINK.sub(lambda m: _rewrite(m.group(0), files), parts[i])
    return "".join(parts)
