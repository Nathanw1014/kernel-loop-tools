#!/usr/bin/env python3
"""lds.py <TYPE> <warptile> [--fp32] [--coopmat] [--bank-offset N] [--mmid] [--limit BYTES]

Static LDS (shared memory) check for a ggml-vulkan mul_mm tile spec, no GPU: prints the bytes the shader would need and exits 1
when they exceed the limit, 0 otherwise. Mirrors ggml_vk_matmul_shmem_support() in ggml/src/ggml-vulkan/ggml-vulkan.cpp (the
float mul_mm path); re-check it against that function when the shader changes, and extend it for your own shader variants.

  TYPE        src0 (weight) ggml type name, e.g. q4_K, iq4_xs (selects the LUT term)
  warptile    the ggml-vulkan warptile vector, comma separated, as in ggml-vulkan.cpp:
                index 0 BLOCK_SIZE, 1 BM, 2 BN, 3 BK, 4 WM, 5 WN, 6 WMITER, 7 TM, 8 TN, 9 TK, 10 WARP (subgroup size)
              only 0, 1, 2, 3, 7, 8 and 10 enter the formula; e.g. 128,128,128,16,64,64,2,4,4,1,32
  --fp32      device without fp16 shared buffers (element size 4 instead of 2)
  --coopmat   cooperative-matrix device: row padding 8 instead of 1, plus the TM x TN x warps f32 staging area
  --bank-offset N   override the row padding (elements)
  --mmid      MUL_MAT_ID variant (adds BN x 2 u16 row ids)
  --limit B   maxComputeSharedMemorySize of your device (vulkaninfo); default 65536

Formula: (BM + BN) x (BK + pad) x elem + [mmid] BN x 4 + [coopmat] TM x TN x (BLOCK_SIZE / WARP) x 4 + LUT(TYPE)."""
import sys

LUT = {'IQ1_S': 2 * 2048, 'IQ1_M': 2 * 2048, 'IQ2_XXS': 8 * 256, 'IQ2_XS': 8 * 512, 'IQ2_S': 8 * 1024,
       'IQ3_XXS': 4 * 256, 'IQ3_S': 4 * 512, 'IQ4_NL': 4 * 16, 'IQ4_XS': 4 * 16, 'MXFP4': 4 * 16, 'NVFP4': 4 * 16 + 256 * 4}

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = int(args[i + 1]); del args[i:i + 2]; return v
    return default
def flag(name):
    if name in args: args.remove(name); return True
    return False
if len(args) < 2 or args[0] in ('-h', '--help'): print(__doc__); sys.exit(2)
limit = opt('--limit', 65536); bank = opt('--bank-offset', None)
fp32 = flag('--fp32'); coopmat = flag('--coopmat'); mmid = flag('--mmid')
if len(args) != 2: print(__doc__); sys.exit(2)
t, spec_s = args[0].upper(), args[1]
w = [int(x) for x in spec_s.split(',')]
if len(w) < 11: sys.exit(f'lds.py: warptile needs at least 11 entries, got {len(w)}')
blk, bm, bn, bk, tm, tn, warp = w[0], w[1], w[2], w[3], w[7], w[8], w[10]
if warp <= 0 or blk % warp: sys.exit('lds.py: BLOCK_SIZE must be a multiple of WARP')
pad = bank if bank is not None else (8 if coopmat else 1)
elem = 4 if fp32 else 2
warps = blk // warp
load_bufs = (bm + bn) * (bk + pad) * elem
row_ids = bn * 2 * 2 if mmid else 0
stage = tm * tn * warps * 4 if coopmat else 0   # upstream: warptile[7] * warptile[8] * warps * sizeof(float)
lut = LUT.get(t, 0)
tot = load_bufs + row_ids + stage + lut
print(f"LDS {t} {spec_s}: load_bufs {load_bufs} + mmid_row_ids {row_ids} + coopmat_stage {stage} + lut {lut} = {tot} B "
      f"(limit {limit}) {'OK' if tot <= limit else 'OVER'}")
sys.exit(0 if tot <= limit else 1)
