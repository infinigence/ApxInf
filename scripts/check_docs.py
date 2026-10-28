#!/usr/bin/env python3
"""Documentation lint for ApxInf. Run from repo root: `just doc-lint`."""
import html
import os
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

errors = []


def err(path, msg, fix):
    errors.append("{0}: {1}\n    fix: {2}".format(path, msg, fix))


# Rule 1: root-level .md whitelist
ALLOWED_ROOT_MD = {
    'README.md', 'AGENTS.md', 'CLAUDE.md', 'REVIEW.md', 'CONTRIBUTING.md',
    'CODE_OF_CONDUCT.md', 'SECURITY.md', 'CHANGELOG.md',
}
for f in os.listdir('.'):
    if f.endswith('.md') and f not in ALLOWED_ROOT_MD:
        err(f, 'markdown file not allowed at repo root',
            'move it into doc/ per doc/README.md#how-docs-are-organized')

# Rule 2: doc/ subdirectory taxonomy
ALLOWED_DOC_TOP = {'README.md', 'ARCHITECTURE.md'}
ALLOWED_DOC_DIRS = {'guides', 'reference', 'models', 'design', 'reports', 'archive'}
for root, dirs, files in os.walk('doc'):
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    for f in files:
        if not f.endswith('.md'):
            continue
        p = os.path.join(root, f)
        rel = os.path.relpath(p, 'doc')
        parts = rel.split(os.sep)
        if len(parts) == 1:
            if parts[0] not in ALLOWED_DOC_TOP:
                err(p, 'doc/<file>.md not in allowed top-level set',
                    'move under one of {0}'.format(sorted(ALLOWED_DOC_DIRS)))
        else:
            if parts[0] not in ALLOWED_DOC_DIRS:
                err(p, 'doc/{0} not an allowed subdirectory'.format(parts[0]),
                    'use one of {0}'.format(sorted(ALLOWED_DOC_DIRS)))

# Rule 3: doc/design/*.md must start with a Status line (first 3 lines)
if os.path.isdir('doc/design'):
    for f in os.listdir('doc/design'):
        if not f.endswith('.md'):
            continue
        p = os.path.join('doc/design', f)
        head = ''.join(open(p).readlines()[:5])
        if not re.search(r'(?i)^#? ?\*?\*?Status\*?\*?\s*[:\-]', head, re.M):
            err(p, 'design doc missing a Status line in its first lines',
                'add "Status: Draft|Accepted|Superseded by <link>|Historical" '
                'right after the title')

# Rule 4: doc/reports/ filename must be YYYY-MM-DD-slug.md and record
# a commit hash + a board name
REPORT_RE = re.compile(r'^\d{4}-\d{2}-\d{2}-[a-z0-9-]+\.md$')
BOARDS = re.compile(
    r'(?i)thor|orin|rtx.?4090|gb10|sm_?8[0-9]|sm_?10[0-9]|sm_?11[0-9]|sm_?12[0-9]|4090')
COMMIT_RE = re.compile(r'\b[0-9a-f]{7,40}\b')
if os.path.isdir('doc/reports'):
    for f in os.listdir('doc/reports'):
        if not f.endswith('.md'):
            continue
        p = os.path.join('doc/reports', f)
        if not REPORT_RE.match(f):
            err(p, 'report filename must be YYYY-MM-DD-<slug>.md',
                'rename per doc/README.md#how-docs-are-organized')
            continue
        body = open(p).read()
        if not COMMIT_RE.search(body):
            err(p, 'report does not name a commit hash',
                'add the baseline commit to the header, e.g. "Baseline commit: abc1234"')
        if not BOARDS.search(body):
            err(p, 'report does not name the board/arch it ran on',
                'add board info, e.g. "Board: Jetson AGX Thor (sm_110)"')

# Rule 5: bare cargo/maturin invocations forbidden in docs outside archive/reports
BARE = re.compile(
    r'^\s*(cargo (build|check|run|clippy|fmt)\b|cargo (test|check) --workspace\b|maturin\b)')
SKIP_DIRS = ('doc/archive', 'doc/reports')
for root, dirs, files in os.walk('doc'):
    if any(root.startswith(d) for d in SKIP_DIRS):
        continue
    for f in files:
        if not f.endswith('.md'):
            continue
        p = os.path.join(root, f)
        in_code = False
        for i, line in enumerate(open(p), 1):
            stripped = line.rstrip()
            if stripped.lstrip().startswith('```'):
                in_code = not in_code
                continue
            if in_code and BARE.search(line):
                err(p, 'line {0}: bare cargo/maturin invocation in docs'.format(i),
                    'replace with the justfile recipe, e.g. `just build-py` '
                    '(doc/README.md#how-docs-are-organized)')
                break

# Include tracked and new, non-ignored Markdown, including agent instructions.
# Git handles build trees, devlocal, vendored submodules and local venvs.
def all_md():
    paths = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'])
    return sorted({os.fsdecode(p) for p in paths.split(b'\0')
                   if p.endswith(b'.md') and os.path.isfile(os.fsdecode(p))})


markdown_files = all_md()

# Rule 6: recipe references (including arguments and fenced examples) exist.
try:
    RECIPES = set(subprocess.check_output(['just', '--summary'], text=True).split())
except (OSError, subprocess.CalledProcessError) as exc:
    err('justfile', 'cannot read recipe list: {0}'.format(exc),
        'install just and ensure just --summary succeeds')
    RECIPES = set()
JUST_REF = re.compile(r'(?:`|^\s*(?:\w+=\S+\s+)*)just ([a-z][a-z0-9-]+)\b')
for p in markdown_files:
    for i, line in enumerate(open(p, errors='replace'), 1):
        for name in JUST_REF.findall(line):
            if name not in RECIPES:
                err(p, 'line {0}: just {1} does not exist'.format(i, name),
                    'add the recipe to the justfile or fix the doc')


# Rule 7: local Markdown links, including same-file and duplicate anchors.
LINK = re.compile(r'\[[^\]]*\]\(([^\s)]*)(?:\s+"[^"\n]*")?\)')


def markdown_anchors(path):
    anchors = set()
    fence = None
    for line in open(path, errors='replace'):
        stripped = line.lstrip()
        marker = re.match(r'(`{3,}|~{3,})', stripped)
        if marker:
            char = marker[1][0]
            if fence is None:
                fence = char
            elif fence == char:
                fence = None
            continue
        if fence:
            continue
        anchors.update(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)', line))
        heading = re.match(r'^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$', line)
        if not heading:
            continue
        title = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', heading[1])
        title = html.unescape(re.sub(r'<[^>]*>', '', title))
        slug = re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-')
        anchor = slug
        suffix = 0
        while anchor in anchors:
            suffix += 1
            anchor = '{0}-{1}'.format(slug, suffix)
        anchors.add(anchor)
    return anchors


anchor_cache = {}
for p in markdown_files:
    for i, line in enumerate(open(p, errors='replace'), 1):
        for target in LINK.findall(line):
            url = urlsplit(target)
            if url.scheme or url.netloc or url.path.startswith('/'):
                continue
            dest = os.path.normpath(os.path.join(os.path.dirname(p), unquote(url.path))) if url.path else p
            if not os.path.exists(dest):
                err(p, 'line {0}: broken link {1}'.format(i, target),
                    'fix the path or remove the link')
            elif url.fragment and dest.endswith('.md'):
                if dest not in anchor_cache:
                    anchor_cache[dest] = markdown_anchors(dest)
                if unquote(url.fragment) not in anchor_cache[dest]:
                    err(p, 'line {0}: missing anchor in {1}'.format(i, target),
                        'link to an existing heading or explicit anchor')

# Rule 8: line budgets
BUDGETS = [
    ('AGENTS.md', 150),
    ('REVIEW.md', 120),
]
for p in markdown_files:
    if os.path.basename(p) == 'AGENTS.md' and p != 'AGENTS.md':
        n = sum(1 for _ in open(p, errors='replace'))
        if n > 60:
            err(p, 'nested AGENTS.md is {0} lines, budget 60'.format(n),
                'keep nested rules to what is unique to this directory')
for p, budget in BUDGETS:
    if os.path.exists(p):
        n = sum(1 for _ in open(p))
        if n > budget:
            err(p, '{0} is {1} lines, budget {2}'.format(p, n, budget),
                'trim per doc/README.md#how-docs-are-organized')
for p in markdown_files:
    if os.path.basename(p) == 'SKILL.md':
        n = sum(1 for _ in open(p, errors='replace'))
        if n > 500:
            err(p, 'SKILL.md is {0} lines, budget 500'.format(n),
                'keep skills short; move detail into linked docs')

# Rule 9: every APXINF_* env var read in source must be registered in
# doc/reference/env-vars.md
env_src = set()
SCAN_DIRS = ('crates', 'src', 'scripts', 'python/apxinf')
for root in SCAN_DIRS:
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ('target', 'node_modules')]
        for f in files:
            if not f.endswith(('.rs', '.py', '.sh', '.cu', '.cpp', '.h', '.cuh')):
                continue
            try:
                body = open(os.path.join(dirpath, f), errors='replace').read()
            except OSError:
                continue
            env_src.update(re.findall(r'\bAPXINF_[A-Z0-9_]+\b', body))
registry_path = 'doc/reference/env-vars.md'
if os.path.exists(registry_path):
    registered = set(re.findall(r'`(APXINF_[A-Z0-9_]+)`',
                                open(registry_path).read()))
    for name in sorted(env_src - registered):
        err(registry_path, 'env var {0} read in source but not registered'.format(name),
            'add it to the appropriate section of doc/reference/env-vars.md')

if not os.path.exists(registry_path):
    err(registry_path, 'environment registry is missing', 'restore the registry')

if errors:
    print('\n'.join(errors))
    print('\ndoc-lint: {0} violation(s)'.format(len(errors)))
    sys.exit(1)
print('doc-lint: 0 violations')
