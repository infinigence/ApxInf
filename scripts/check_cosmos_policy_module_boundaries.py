#!/usr/bin/env python3
"""Guard Cosmos Policy ownership; numerical and device tests remain separate."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / 'crates/apxinf-model/src/cuda_cosmos_policy'


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
    required = {'mod.rs', 'backend.rs', 'config.rs', 'load.rs',
                'model/mod.rs', 'model/blocks/mod.rs', 'model/blocks/bf16.rs',
                'weights/mod.rs', 'weights/host.rs', 'weights/bf16.rs',
                'model_runner/mod.rs', 'model_runner/runner.rs', 'model_runner/prepare.rs'}
    actual = {str(p.relative_to(root)) for p in root.rglob('*.rs')}
    if actual != required:
        errors.append(f'family layout: missing={sorted(required-actual)}, unexpected={sorted(actual-required)}')
        return errors
    execution = r'\b(ExecutionSession|CapturedGraph|PreparedInference|ExecutionPolicy|VlaRequest|InferenceSpec)\b'
    for p in root.rglob('*.rs'):
        text = production(p)
        name = str(p.relative_to(root))
        if re.search(r'\bunsafe\b|::ffi\b', text):
            errors.append(f'{name}: use safe backend APIs')
        if p.name != 'backend.rs' and re.search(r'\bapxinf_cuda(?:_new)?::', text):
            errors.append(f'{name}: import CUDA through backend.rs')
        if name.startswith(('model/', 'weights/')) and re.search(execution + r'|\bmodel_runner\b', text):
            errors.append(f'{name}: computation/weights depend on execution ownership')
        if name.startswith('weights/') and re.search(r'\b(model|blocks)::|\bfn\s+(forward|denoise_step)\b', text):
            errors.append(f'{name}: weights own activation computation')
        if name.startswith('model/') and re.search(r'\bimpl\s+\w*Weights\b|\b\w*Weights::load\b', text):
            errors.append(f'{name}: model owns checkpoint construction')
    model = production(root/'model/mod.rs')
    if re.search(r'\b(DType|Shape|Dit|VaeChunk|ConvWorkspace|RecurrenceCoefficients)\b|\b(alloc|gemm|attention|cast|gather|multiply)\s*\(', model):
        errors.append('model/mod.rs: tensor representations and operators belong to Bf16Blocks')
    backend = production(root/'backend.rs')
    if re.search(r'\b(fn|impl|struct|enum)\b', backend):
        errors.append('backend.rs: only safe API reexports belong here')
    host = production(root/'weights/host.rs')
    if re.search(r'\b(backend|CudaBuffer|CudaContext|Device|upload)\b', host):
        errors.append('weights/host.rs: host storage must not own device representation')
    for name in ['model_runner/runner.rs', 'model_runner/prepare.rs']:
        if re.search(r'\b(blocks|Dit|VaeChunk|ConvWorkspace)\b', production(root/name)):
            errors.append(f'{name}: runner must use the model interface')
    return errors


if __name__ == '__main__':
    errors = check(ROOT)
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        sys.exit(1)
    print('Cosmos Policy module ownership checks passed')
