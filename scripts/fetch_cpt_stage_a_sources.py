"""Archive the public SC release for Stage A provenance inspection.

Other primary pages are archived separately as web-reader responses. This fetch
does not change the frozen evaluation cases or publish downloaded content.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

DEST = Path('CPT_resources/llin-stage-a-20260920-02')
URLS = {
    'sc_release': 'https://codeload.github.com/Damon-GSY/SC-bench/zip/refs/heads/main',
}

if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = []
    for identity, url in URLS.items():
        req = urllib.request.Request(url, headers={'User-Agent': 'source-audit/1.0'})
        with urllib.request.urlopen(req, timeout=40) as response:
            body, resolved = response.read(), response.url
            content_type = response.headers.get_content_type()
        suffix = '.pdf' if body.startswith(b'%PDF') else '.zip' if identity == 'sc_release' else '.html'
        path = DEST / (identity + '.private' + suffix)
        path.write_bytes(body)
        manifest.append(dict(id=identity, url=url, resolved_url=resolved, file=path.name,
                             content_type=content_type, sha256=hashlib.sha256(body).hexdigest(), bytes=len(body)))
        (DEST / 'downloads.private.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        print(identity, len(body), content_type, flush=True)
