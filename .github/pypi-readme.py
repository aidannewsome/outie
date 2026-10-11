# Points the README's relative links at this release on GitHub, as PyPI shows the README with nothing to resolve them against.
# Run before each build, so every release's page links the figures and files it was made with.
import os
import re
from pathlib import Path

repo, ref = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_REF_NAME"]
link = re.compile(r"(!?)\[([^\]]*)\]\((?![a-z]+:|#|/)([^)]+)\)")


def absolute(match):
    image, text, path = match.groups()
    base = f"https://raw.githubusercontent.com/{repo}/{ref}" if image else f"https://github.com/{repo}/blob/{ref}"
    return f"{image}[{text}]({base}/{path})"


readme = Path("README.md")
readme.write_text(link.sub(absolute, readme.read_text(encoding="utf-8")), encoding="utf-8")
