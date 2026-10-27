#!/usr/bin/env python3
"""Reject broken repository links and links into ignored rendering scratch."""
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    errors = []
    files = [root / 'README.md', *sorted((root / 'docs').rglob('*.md')),
             *sorted((root / 'benchmarks').rglob('*.md'))]
    for path in files:
        for match in re.finditer(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            url = match[1]
            if re.match(r'[a-zA-Z][a-zA-Z0-9+.-]*:', url) or url.startswith('#'):
                continue
            target = (path.parent / unquote(url.split('#')[0])).resolve()
            if target.is_relative_to(root / 'out') or not target.exists():
                errors.append(f'{path.relative_to(root)}: unavailable repository link {url}')
    return errors


if __name__ == '__main__':
    errors = check()
    print('\n'.join(errors) if errors else 'Repository documentation links OK')
    sys.exit(bool(errors))
