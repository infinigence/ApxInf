#!/usr/bin/env python3
"""Measure Qwen-Drive resident-image request latency with constructed input."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from _benchmark import provenance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python/apxinf'))


def synthetic_observation(seed=0):
    from apxinf.policies.impls.qwen_drive import CAMERA_VIEWS
    rng = np.random.default_rng(seed)
    views = {}
    for camera in CAMERA_VIEWS:
        views[camera] = [
            {'image': rng.integers(0, 256, (900, 1600, 3), dtype=np.uint8),
             'target_size': (384, 416) if frame < 3 else (720, 799)}
            for frame in range(4)
        ]
    history = np.zeros((16, 3), dtype=np.float32)
    history[:, 0] = np.linspace(-7.5, 0, 16)
    velocity = np.zeros((16, 2), dtype=np.float32)
    velocity[:, 0] = 5
    observation = dict(views=views, history=history, history_velocity=velocity,
                       history_acceleration=np.zeros((16, 2), dtype=np.float32),
                       ego_velocity=[5., 0.], ego_acceleration=[0., 0.],
                       driving_command=[0., 0., 1., 0.], nav_command=2)
    noise = rng.standard_normal((1, 50, 3), dtype=np.float32)
    return observation, noise


def main(policy_loader=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--precision', choices=['bf16'], default='bf16')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--samples', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tactics', type=Path)
    p.add_argument('--out', type=Path, default=Path('devlocal/qwen-drive-bench/results/latency.json'))
    a = p.parse_args()
    if a.warmup < 1 or a.samples < 1:
        p.error('--warmup and --samples must be positive')
    from apxinf import AutoPolicy
    options = dict(model_type='qwen_drive', model_variant=a.precision,
                   device=a.device, mode='direct_planning', num_steps=10,
                   planner=a.model_dir / 'planner-sft', tactics=a.tactics)
    policy = (policy_loader or AutoPolicy.from_pretrained)(a.model_dir, **options)
    try:
        observation, noise = synthetic_observation(a.seed)
        for _ in range(a.warmup):
            policy.infer(observation, noise=noise)
        samples = []
        first = None
        for _ in range(a.samples):
            start = time.perf_counter()
            result = policy.infer(observation, noise=noise)
            samples.append((time.perf_counter() - start) * 1000)
            actions = np.asarray(result['actions']).copy()
            if actions.shape != (50, 3) or not np.isfinite(actions).all():
                raise ValueError(f'invalid trajectory: {actions.shape}')
            if first is None:
                first = actions
            elif not np.array_equal(first, actions):
                raise ValueError('identical input produced different trajectories')
        report = dict(schema='apxinf.qwen-drive.benchmark.v1', input_source='synthetic',
                      input_profile='three-cameras-four-frames-3387-capacity', seed=a.seed,
                      model_dir=str(a.model_dir), precision=a.precision,
                      tactics=str(a.tactics) if a.tactics else None,
                      timing_boundary='resident RGB through policy and trajectory D2H',
                      warmup=a.warmup, samples_ms=samples,
                      p50_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)),
                      output_shape=list(first.shape), output_sha256=hashlib.sha256(first.tobytes()).hexdigest())
        report["provenance"] = provenance(ROOT, a.model_dir, tactics=a.tactics)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        policy.close()


if __name__ == '__main__':
    main()
