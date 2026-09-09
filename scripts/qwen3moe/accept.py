#!/usr/bin/env python3
"""Thor-U acceptance: build, teacher-forced checks, CLI integrity, ISL/OSL sweep.

Source /opt/data/dev/env.sh first. Every invocation leaves a unique artifact
directory and appends a JSONL record, including failed runs. Baseline numerical
envelope is explicit; performance milestones are optional hard gates.
"""
import argparse
import datetime
import filecmp
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from source_snapshot import capture_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/opt/data/models/Qwen3-30B-A3B-Instruct-2507-AWQ')
    parser.add_argument('--reference', type=Path, default=Path('/opt/data/dev/ref'))
    parser.add_argument('--output', type=Path, default=Path('/opt/data/dev/acceptance'))
    parser.add_argument('--cases', default='raw0,chat0', help='comma-separated reference fixture names')
    parser.add_argument('--verify-only', action='store_true', help='run numerical checks only; cannot close a performance milestone')
    parser.add_argument('--isl', default='128,1024,4096')
    parser.add_argument('--osl', default='128')
    parser.add_argument('--previous', type=Path, help='require bit-identical logits to this run directory')
    parser.add_argument('--skip-build', action='store_true', help='use existing binaries; record their hashes and mark build as external')
    parser.add_argument('--milestone', choices=['baseline', 'M1', 'M2', 'M3', 'M4'], default='baseline')
    args = parser.parse_args()
    if args.verify_only and args.milestone != 'baseline':
        parser.error('--verify-only cannot be combined with a performance milestone')
    cases = args.cases.split(',')
    if not cases or any(not name or not all(c.isalnum() or c in '_-' for c in name) for name in cases):
        parser.error('--cases must contain simple fixture names')
    repo = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run = args.output.resolve() / (stamp + '-' + revision[:8])
    run.mkdir(parents=True)
    record = dict(revision=revision, started=stamp, artifacts=str(run), milestone=args.milestone,
                  environment={k: v for k, v in os.environ.items() if k.startswith('APXINF_')},
                  status='failed', scope='numerical only' if args.verify_only else 'full acceptance', commands=[])
    target = Path(os.environ.get('CARGO_TARGET_DIR', str(repo / 'target'))).resolve() / 'release'

    def command(name, argv):
        started = time.monotonic()
        with (run / (name + '.log')).open('w') as log:
            result = subprocess.run([str(x) for x in argv], cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        record['commands'].append(dict(name=name, argv=[str(x) for x in argv],
                                       returncode=result.returncode, seconds=time.monotonic()-started))
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {run / (name + ".log")}')

    try:
        record.update(capture_source(repo, run, args.output))
        if not args.skip_build:
            command('build', ['cargo', 'build', '--release', '--features', 'cuda', '--bin', 'apxinf', '--example', 'qwen3moe_verify'])
        record['build'] = 'external' if args.skip_build else 'built in this run'
        record['binaries'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [target / 'apxinf', target / 'examples/qwen3moe_verify']}
        for case in cases:
            command(case, [target / 'examples/qwen3moe_verify', '--model', args.model,
                           '--case', args.reference / (case + '.json'), '--out', run / case])
            command(case + '-compare', [sys.executable, Path(__file__).with_name('compare_reference.py'),
                    '--reference', args.reference / (case + '.npz'), '--apxinf', run / (case + '.json'), '--strict']
                    + (['--allow-tie-step', '2'] if case == 'raw0' else []))
            if args.previous and not filecmp.cmp(run / (case + '.bin'), args.previous / (case + '.bin'), shallow=False):
                raise RuntimeError(f'{case}: logits are not bit-identical to {args.previous}')
        if not args.verify_only:
            command('generate', [target / 'apxinf', 'generate', '--model', args.model, '--device', 'cuda',
                    '--dtype', 'auto', '--greedy', '--prompt', 'Give me a short introduction to large language models.', '--max-tokens', '64'])
            command('bench', [target / 'apxinf', 'bench', '--model', args.model, '--device', 'cuda', '--dtype', 'auto',
                    '--isl', args.isl, '--osl', args.osl, '--warmup', '1', '--iters', '3', '--json', run / 'bench.json'])
            record['benchmark'] = json.loads((run / 'bench.json').read_text())
        if args.milestone in ['M1', 'M2', 'M3']:
            row = next(r for r in record['benchmark']['rows'] if r['isl'] == 1024 and r['osl'] == 128)
            if row['prefill_tokens_per_s'] < 3000:
                raise RuntimeError('M1 prefill gate not met')
            required = {'M1': 0, 'M2': 70, 'M3': 100}[args.milestone]
            if row['decode_tokens_per_s'] < required:
                raise RuntimeError(f'{args.milestone} decode gate not met')
        if args.milestone == 'M4':
            raise RuntimeError('M4 additionally requires long-context reference and workspace evidence; review before closure')
        record['status'] = 'passed'
    except Exception as error:
        record['error'] = str(error)
    finally:
        (run / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        with (args.output / 'results.jsonl').open('a') as log:
            log.write(json.dumps(record) + '\n')
        print(json.dumps(record, indent=2))
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
