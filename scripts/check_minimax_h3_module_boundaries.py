#!/usr/bin/env python3
"""Guard MiniMax-H3 (video and audio) ownership; numerical and device tests remain separate."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / 'crates/apxinf-model/src/cuda_minimax_h3'


def production(path):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', path.read_text(), flags=re.S)
    while match := re.search(r'#\[cfg\(test\)\]\s*mod\s+\w+\s*\{', text):
        end, depth = match.end(), 1
        while depth and end < len(text):
            depth += (text[end] == '{') - (text[end] == '}')
            end += 1
        text = text[:match.start()] + text[end:]
    return text


def check(root):
    errors = []
    required = {'mod.rs', 'backend.rs', 'config.rs', 'load.rs', 'package.rs',
                'model/mod.rs', 'model/host.rs', 'model/layout.rs', 'model/noise.rs', 'model/vae_host.rs',
                'model/blocks/mod.rs', 'model/blocks/util.rs', 'model/blocks/int8.rs', 'model/blocks/refiner.rs',
                'model/blocks/text.rs', 'model/blocks/vae.rs', 'model/blocks/audio_vae.rs', 'tests/mod.rs', 'tests/parity.rs',
                'weights/mod.rs', 'weights/host.rs', 'weights/bf16.rs', 'weights/text.rs', 'weights/vae.rs', 'weights/audio_vae.rs',
                'model_runner/mod.rs', 'model_runner/runner.rs', 'model_runner/prepare.rs', 'model_runner/video.rs',
                'model_runner/audio.rs'}
    actual = {str(p.relative_to(root)) for p in root.rglob('*.rs')}
    if actual != required:
        errors.append(f'family layout: missing={sorted(required-actual)}, unexpected={sorted(actual-required)}')
        return errors
    execution = r'\b(ExecutionSession|CapturedGraph|PreparedPhase|VideoRuntime|VideoRequest|GeneratedVideo|LoadOptions)\b'
    for p in root.rglob('*.rs'):
        text = production(p)
        name = str(p.relative_to(root))
        if re.search(r'\bunsafe\b|::ffi\b', text):
            errors.append(f'{name}: use safe backend APIs')
        if p.name != 'backend.rs' and re.search(r'\bapxinf_cuda(?:_new)?::', text):
            errors.append(f'{name}: import CUDA through backend.rs')
        if name.startswith(('model/', 'weights/')) and re.search(execution + r'|\bmodel_runner\b', text):
            errors.append(f'{name}: computation/weights depend on request or execution ownership')
        if name.startswith('weights/') and re.search(r'\b(model|blocks)::|\bfn\s+(forward|decode|denoise_step)\b', text):
            errors.append(f'{name}: weights own activation computation')
        if name.startswith('model/') and re.search(r'\bimpl\s+\w*Weights\b|\b\w*Weights::load\b', text):
            errors.append(f'{name}: model owns checkpoint construction')
        if name.startswith('model/') and not name.startswith('model/blocks/') and re.search(r'\bCuda(Buffer|Context)\b|\bcrate::[a-z_]*::backend\b', text):
            errors.append(f'{name}: host arithmetic modules must not name device types; tensors belong to model/blocks')
        if name.startswith('model/blocks/') and re.search(r'\bTokenizer\b|\bstd::fs\b|\bFile::open\b', text):
            errors.append(f'{name}: blocks take prepared tensors, not files or tokenizers')
    backend = production(root / 'backend.rs')
    if re.search(r'\b(fn|impl|struct|enum)\b', backend):
        errors.append('backend.rs: only safe API reexports belong here')
    host = production(root / 'weights/host.rs')
    if re.search(r'\b(backend|CudaBuffer|CudaContext|Device|upload)\b', host):
        errors.append('weights/host.rs: host storage must not own device representation')
    for name in ['model_runner/runner.rs', 'model_runner/prepare.rs', 'model_runner/video.rs', 'model_runner/audio.rs']:
        if re.search(r'\b(Int8Linear|DenseLinear|VvLinear|AudioConv|RepackedWeight|GemmArgs|AttentionArgs)\b', production(root / name)):
            errors.append(f'{name}: the runner sequences stages through the blocks and never calls operators on weights')
    package = production(root / 'package.rs')
    if re.search(r'\bbackend\b|\bCuda', package):
        errors.append('package.rs: the manifest reader is device independent')
    return errors


if __name__ == '__main__':
    errors = check(ROOT)
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        sys.exit(1)
    print('MiniMax-H3 module ownership checks passed')
