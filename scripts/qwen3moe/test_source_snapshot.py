"""Source evidence must survive edits after acceptance, including new kernels."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from source_snapshot import capture_source


class SourceSnapshot(unittest.TestCase):
    def test_changes_survive_later_edits(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1',
        }):
            root = Path(directory)
            repo = root / 'repo'
            repo.mkdir()
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=repo, stderr=subprocess.DEVNULL)
            git('init')
            (repo / 'tracked.cu').write_text('original\n')
            (repo / 'removed.cu').write_text('removed\n')
            git('add', '.')
            git('-c', 'user.name=Snapshot Test', '-c', 'user.email=snapshot@example.invalid',
                'commit', '-m', 'fixture')
            (repo / 'tracked.cu').write_text('changed\n')
            (repo / 'new.cuh').write_bytes(b'new kernel\x00bytes\n')
            (repo / 'removed.cu').unlink()
            outside = root / 'outside'
            outside.write_text('must not enter the source archive')
            (repo / 'link').symlink_to(outside)
            output = repo / 'acceptance'
            run = output / 'run'
            run.mkdir(parents=True)
            (run / 'existing.log').write_text('not source')
            result = capture_source(repo, run, output)
            (repo / 'tracked.cu').write_text('later edit\n')
            (repo / 'new.cuh').unlink()
            state = result['source_snapshot']
            archive = run / state['archive']
            self.assertEqual(state['sha256'], hashlib.sha256(archive.read_bytes()).hexdigest())
            self.assertEqual(state['removed'], ['removed.cu'])
            with tarfile.open(archive) as tar:
                self.assertEqual(set(tar.getnames()), {'tracked.cu', 'new.cuh', 'link'})
                self.assertEqual(tar.extractfile('tracked.cu').read(), b'changed\n')
                self.assertEqual(tar.extractfile('new.cuh').read(), b'new kernel\x00bytes\n')
                self.assertTrue(tar.getmember('link').issym())
                self.assertEqual(tar.getmember('link').size, 0)
            manifest = json.loads((run / 'source-sha256.json').read_text())
            self.assertEqual(manifest['new.cuh'], hashlib.sha256(b'new kernel\x00bytes\n').hexdigest())
            self.assertFalse(any(name.startswith('acceptance/') for name in manifest))
            self.assertIn(b'deleted file', (run / 'source.patch').read_bytes())

    def test_output_cannot_contain_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / 'repo'
            repo.mkdir()
            with self.assertRaisesRegex(ValueError, 'must not contain'):
                capture_source(repo, root / 'run', root)


if __name__ == '__main__':
    unittest.main()
