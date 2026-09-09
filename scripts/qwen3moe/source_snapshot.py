"""Retain reproducible source changes beside each acceptance result."""
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile


def capture_source(repo, run, output_root):
    repo, run, output_root = map(lambda p: Path(p).resolve(), (repo, run, output_root))
    if output_root == repo or repo.is_relative_to(output_root):
        raise ValueError('acceptance output must not contain the source checkout')

    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo)

    def names(*args):
        return {os.fsdecode(p) for p in git(*args).split(b'\0') if p}

    changed = names('diff', 'HEAD', '--name-only', '-z')
    untracked = names('ls-files', '--others', '--exclude-standard', '-z')
    paths = names('ls-files', '-co', '--exclude-standard', '-z')
    changed |= untracked
    (run / 'source.patch').write_bytes(git('diff', '--binary', 'HEAD'))
    manifest, links, removed = {}, {}, []
    archive = run / 'source-changes.tar.gz'
    with tarfile.open(archive, 'w:gz', dereference=False) as tar:
        for name in sorted(paths):
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'unsafe source path: {name!r}')
            path = repo / relative
            # An output directory inside the checkout must not capture itself
            # or earlier acceptance dumps as untracked "source".
            if path.is_relative_to(output_root):
                continue
            try:
                info = path.lstat()
            except FileNotFoundError:
                if name in changed:
                    removed.append(name)
                continue
            member = tarfile.TarInfo(name)
            member.mode = stat.S_IMODE(info.st_mode)
            member.mtime = int(info.st_mtime)
            if stat.S_ISLNK(info.st_mode):
                target = os.readlink(path)
                links[name] = target
                manifest[name] = hashlib.sha256(os.fsencode(target)).hexdigest()
                if name in changed:
                    member.type = tarfile.SYMTYPE
                    member.linkname = target
                    tar.addfile(member)
            elif stat.S_ISREG(info.st_mode):
                # Hash and archive the same read, retaining untracked kernel
                # contents even after the working tree changes again.
                data = path.read_bytes()
                manifest[name] = hashlib.sha256(data).hexdigest()
                if name in changed:
                    member.size = len(data)
                    tar.addfile(member, io.BytesIO(data))
            else:
                raise ValueError(f'unsupported source file type: {name!r}')
    (run / 'source-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    state = dict(archive=archive.name, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                 removed=removed, symlinks=links)
    (run / 'source-state.json').write_text(json.dumps(state, indent=2) + '\n')
    return dict(git_status=git('status', '--porcelain').decode(), source_snapshot=state)
