#!/usr/bin/env python3
"""Documentation lint for ApxInf. Run from repo root: `just doc-lint`."""
import os
import re
import subprocess
import sys

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

# Rule 6: `just <recipe>` references must exist in the justfile
try:
    out = subprocess.check_output(['just', '--summary']).decode()
    RECIPES = set(out.split())
except Exception:
    RECIPES = None
if RECIPES is not None:
    JUST_REF = re.compile(r'`just ([a-z][a-z0-9-]+)`')
    seen = set()
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in
                   ('target', 'devlocal', '.git', 'node_modules', '.venv',
                    '.github', 'hooks', '.claude', '.agents')]
        for f in files:
            if not f.endswith('.md'):
                continue
            p = os.path.join(root, f).lstrip('./')
            for i, line in enumerate(open(p, errors='replace'), 1):
                for m in JUST_REF.finditer(line):
                    name = m.group(1)
                    if name not in RECIPES and name != '--list':
                        err(p, 'line {0}: `just {1}` does not exist'.format(i, name),
                            'add the recipe to the justfile or fix the doc')

# Rule 7: relative markdown links must resolve (files; anchor check TODO)
LINK = re.compile(r'\[[^\]]*\]\((?!https?://)([^)#\s]+)(#[^)]*)?\)')
def all_md():
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in
                   ('target', 'devlocal', '.git', 'node_modules', '.venv',
                    '.github', 'hooks', '.claude', '.agents')]
        for f in files:
            if f.endswith('.md'):
                yield os.path.join(root, f).lstrip('./')

for p in all_md():
    base = os.path.dirname(p)
    for i, line in enumerate(open(p, errors='replace'), 1):
        for m in LINK.finditer(line):
            target = m.group(1)
            if target.startswith('/'):
                continue
            dest = os.path.normpath(os.path.join(base, target))
            if not os.path.exists(dest):
                err(p, 'line {0}: broken link {1}'.format(i, target),
                    'fix the path or remove the link')

# Rule 8: line budgets
BUDGETS = [
    ('AGENTS.md', 150),
    ('REVIEW.md', 120),
]
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in
               ('target', 'devlocal', '.git', 'node_modules', '.venv')]
    for f in files:
        p = os.path.join(root, f).lstrip('./')
        if f == 'AGENTS.md' and p != 'AGENTS.md':
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
for root, dirs, files in os.walk('.agents'):
    for f in files:
        if f == 'SKILL.md':
            p = os.path.join(root, f).lstrip('./')
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

if errors:
    print('\n'.join(errors))
    print('\ndoc-lint: {0} violation(s)'.format(len(errors)))
    sys.exit(1)
print('doc-lint: 0 violations')
