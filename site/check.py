"""Check the entire production artifact under a GitHub Pages repository prefix."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

DIST = Path(__file__).resolve().parent / "dist"
class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        self.links += [v for k, v in attrs if k in ("href", "src") and v]

errors, count = [], 0
for page in DIST.rglob("*.html"):
    parser = Links()
    parser.feed(page.read_text())
    for link in parser.links:
        url = urlsplit(link)
        if url.scheme or url.netloc or not url.path:
            continue
        if url.path.startswith("/"):
            errors.append(f"{page.name}: root absolute path breaks /deepinfer-mac/: {link}")
        target = (page.parent / unquote(url.path)).resolve()
        if not target.is_relative_to(DIST.resolve()) or not target.exists():
            errors.append(f"{page.name}: missing or escaping asset: {link}")
        count += 1
assert not errors, "\n".join(errors)
assert len(list(DIST.glob("*.html"))) == 7
assert (DIST / ".nojekyll").exists()
assert not list(DIST.rglob("*.map"))
print(f"PASS: {count} local resource/document links, 7 pages, relocatable /deepinfer-mac/ artifact")
