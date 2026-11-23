"""Export MP22 W8A8 NN/NT decode GEMMs from a read-only ComputeAsmKern checkout.

Requires the renderer's Python dependencies and the MUSA assembler/linker.
Generated templates, images and provenance belong in an explicit private output
folder; promote only the reviewed .asm files into kernels/asm/.
The qint8_gemm_nn/nt_32x128 image names identify their GEMM template origin.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--renderer', type=Path, required=True, help='ComputeAsmKern checkout')
parser.add_argument('--out', type=Path, required=True, help='private output directory')
parser.add_argument('--musa-path', type=Path, default=Path('/usr/local/musa'))
args = parser.parse_args()
SOURCE = args.renderer.resolve()
if not (SOURCE / 'src/asm_kernel/musa_hgemm_32x128.s').is_file():
    parser.error('renderer has no src/asm_kernel/musa_hgemm_32x128.s')
sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE / 'src/asm_kernel')]
from asm_render.asm_render import render_file

original = SOURCE / 'src/asm_kernel/musa_hgemm_32x128.s'
expected_sha256 = '3ca68d1eb983024d460f161f05f8a79abbe5b87240257bad1ae91bdf13381732'
if hashlib.sha256(original.read_bytes()).hexdigest() != expected_sha256:
    parser.error('unreviewed template revision; use the source recorded in kernels/asm/README.md')
text = original.read_text()
# The upstream template parameterizes INT8 loads but hardcodes float MMA
# accumulators and derives BF16 stores from the input rather than output type.
text = text.replace('].f32', '].{compute_suffix}')
# The upstream NN store mapping handles FP16/TF32 but sends INT8 through
# the TF32 branch. INT8 has eight 16-byte segments per 128-column row.
text = text.replace('b_r4_thread = 16 if input_dtype_size == 2 else 32',
                    'b_r4_thread = nr_n_block // (16 // input_dtype_size)')
text = text.replace('(c_dtype_size == 2 and mma_dtype == Dtype.bf16)',
                    '(c_dtype == Dtype.bf16)')
text = text.replace('s:0-3  : pc_temp, remain_k, tile_m, tile_n',
                    '155-158: row_scale<0-3>\n159-166: column_scale<0-7>\n'
                    '167-178: scale_offset<0-11>\ns:0-3  : pc_temp, remain_k, tile_m, tile_n')
# At this point all accumulators and the MMA store row/column indices are
# available; temporary scaling registers are disjoint from the store epilogue.
anchor = '<CODE>\n# store stage\nrst = []'
assert text.count(anchor) == 1
dequant = '''<CODE>
rst = ['[NwwDwww] CTRL.NOP']
fc = AutoNFence('sw')
for row in range(4):
  rst.append(f'[N{fc()}D---] INT.ADD scale_offset{row}.s32, stg_m_idx.s32, ConstI{row * 8}.s32')
  rst.append(f'[N{fc()}D---] BIT.ROT.LSL scale_offset{row}, ConstI1, scale_offset{row}')
  rst.append(f'[NwwD-s-] DMA.LD.B16 row_scale{row}.l, input_a_scales[0:1], scale_offset{row}, _, _, _, _, input_a_scales[0:2]')
for column in range(8):
  offset = (column // 2) * 32 + (column % 2) * 16
  rst.append(f'[N{fc()}D---] INT.ADD scale_offset{column+4}.s32, stg_n_idx.s32, ConstI{offset}.s32')
  rst.append(f'[N{fc()}D---] BIT.ROT.LSL scale_offset{column+4}, ConstI1, scale_offset{column+4}')
  rst.append(f'[NwwD-s-] DMA.LD.B16 column_scale{column}.l, input_b_scales[0:1], scale_offset{column+4}, _, _, _, _, input_b_scales[0:2]')
rst.append('[NwwDwww] CTRL.NOP')
for row in range(4):
  for nid in range(4):
    for half in range(2):
      reg = f'cm0n{nid}x{row * 2 + half}'
      rst.append(f'[N{fc()}D---] MOV.S32_TO_F32_RTE {reg}, {reg}')
      rst.append(f'[N{fc()}D---] FOP.MUL {reg}, {reg}, column_scale{nid * 2 + half}.l')
      rst.append(f'[N{fc()}D---] FOP.MUL {reg}, {reg}, row_scale{row}.l')
rst.append('[NwwD---] CTRL.NOP')
code = '\\n'.join(rst)
</CODE>
'''
text = text.replace(anchor, dequant + '\n' + anchor)
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
template = out / 'musa_w8a8_decode_32x128.s'
template.write_text(text)
manifest = []
for nt in [False, True]:
  cfg = dict(trans_a=False, trans_b=nt, mma_dtype='int8', b_dtype='int8',
             c_dtype='bfloat16', large_load=False, offline_reorder=False,
             online_reorder=False)
  assembly, *_ = render_file(str(template), cfg, kern_name_prefix='apxinf_',
                             arch='QY2', static_check_assert=True)
  assembly = ''.join(assembly)
  symbol = f'apxinf_bf16s8gemm_n{"t" if nt else "n"}_tc_128_32x128_epilogue'
  path = out / f'qint8_gemm_n{"t" if nt else "n"}_32x128.asm'
  path.write_text(assembly)
  obj = path.with_suffix('.o')
  image = path.with_suffix('.mubin')
  subprocess.run([str(args.musa_path / 'bin/llvm-mc'), '--arch=mtgpu', '--filetype=obj',
                  '-mcpu=mp_22', str(path), '-o', str(obj)], check=True)
  subprocess.run([str(args.musa_path / 'bin/lld'), '-flavor', 'gnu', '--no-undefined',
                  '-shared', str(obj), '-o', str(image)], check=True)
  manifest.append(dict(name='nt' if nt else 'nn', path=str(image.resolve()),
                       symbol=symbol, tile_m=32, tile_n=128, threads=128,
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(out / 'provenance.json').write_text(json.dumps(dict(
  source=str(original), source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
  source_revision=subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip(),
), indent=2) + '\n')
print('Built NN and NT INT8 decode images')
