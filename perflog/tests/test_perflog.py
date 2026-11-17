#!/usr/bin/env python3
"""Run every perflog.py subcommand on the synthetic fixtures here and check hand-computed numbers.
usage: python3 perflog/tests/test_perflog.py   (exit 0 = all pass)"""
import pathlib, re, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
TOOL = HERE.parent / 'perflog.py'
S, C, CUT, CSV, CLS = (str(HERE / f) for f in ('serial.log', 'concurrent.log', 'cut.log', 'kernel_stats.csv', 'classes.txt'))

# (args, substrings that must appear in stdout; runs of spaces compare equal)
CASES = [
    (['blocks', S], ['5 blocks, serialising logger', '5000.020', 'ubatch cycle: 2 block(s) (exact)', '[5, 3, 5, 3]']),
    (['top', S], ['block 4 of 5', 'row sum 0.515 ms vs logged Total time 0.515 ms (+0.000%)', 'shown 0.515 of 0.515 ms']),
    (['top', S, '--block', 'largest'], ['block 0 of 5', 'first graph']),
    (['top', S, '--ubatch', '--by', 'class'], ['last 2 of 5 blocks', 'moe_gemm            0.300 ms',
                                               'dense_mm            1.210 ms', 'attn                0.150 ms',
                                               'norm                0.050 ms', 'glue                0.005 ms',
                                               'matmul share (moe_gemm + dense_mm): 88.0%']),
    (['top', S, '--ubatch', '--by', 'class', '--classes', CLS], ['gemm                1.510 ms', 'rest                0.055 ms']),
    (['top', S, '--by', 'kind', '--all'], ['MUL_MAT_VEC', 'shown 5003.440 of 5003.440 ms']),
    (['top', CUT], ['block 0 of 1', 'row sum 0.042 ms (no logged Total time)']),
    (['sum', S], ['blocks 5, total 5003.4 ms', 'logged Total time sum 5003.4 ms (5 blocks)']),
    (['sum', S, C], ['label | A us/block', 'GET_ROWS | 2.0 | 5.0 | +3.0']),
    (['shapes', S, '--ubatch'], ['MUL_MAT q4_K m=4096 n=512 k=4096', '500.0   34.36       34.36',
                                 '210.0    1.25        1.25', 'total 1.210 ms in 2 shapes']),
    (['shapes', S, '--all'], ['833666.7 0.02 0.02', '205.0 1.28 1.28']),
    (['shapes', S, C, '--block', 'last'], ['dense GEMM per shape (A vs B']),
    (['shapes', C], ['concurrent logger', '1.200 ms of GEMM time sits in multi-op rows', 'total 0.000 ms in 0 shapes']),
    (['diff', S, C, '--block', 'last', '--labels', 'ser', 'con'],
     ['op time ser 0.515 ms, con 1.405 ms, diff -0.890 ms', 'ADD, RMS_NORM_MUL', 'GEMM time in multi-op rows ser 0.000 ms, con 1.200 ms']),
    (['ubatch', S], ['last 2 of 5 blocks (cycle exact), 8 op calls, 1.715 ms op time',
                     '2.000 ms/ubatch; from avg_ts: 2.000 ms/ubatch', 'residual (host / unlogged / overlap) +0.285 ms']),
    (['ubatch', S, S, '--tg', '400', '--tg', '500', '--shapes'],
     ['wall: 2.500 ms/ubatch (--tg 400.0)', 'residual (host / unlogged / overlap) +0.785 ms',
      'wall A vs B: +0.500 ms (+25.0%); op time +0.000 ms; residual +0.500 ms']),
    (['xbackend', '--vk', S, '--hip', CSV], ['-- vulkan: 0.515 ms', '-- hip: 4.810 ms', 'dense_mm   83.2%      4.000 ms',
                                              'gemm_prep   8.3%      0.400 ms']),
]


def main():
    fails = 0
    for args, want in CASES:
        r = subprocess.run([sys.executable, str(TOOL)] + args, capture_output=True, text=True)
        sq = lambda x: re.sub(r' +', ' ', x)
        missing = [w for w in want if sq(w) not in sq(r.stdout)]
        ok = r.returncode == 0 and not missing
        fails += not ok
        print(f'{"ok  " if ok else "FAIL"} {" ".join(args[:1] + [pathlib.Path(a).name for a in args[1:]])}')
        if not ok:
            print(f'  rc={r.returncode} missing={missing}\n  stderr: {r.stderr.strip()[:400]}')
            print('  ' + r.stdout.replace('\n', '\n  '))
    print(f'{len(CASES) - fails}/{len(CASES)} passed')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
