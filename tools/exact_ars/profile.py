#!/usr/bin/env python3
"""Compare ordinary ARS batches with ad17c90 and isolate hot-path costs.

Uses only standard Python, Git and Clang (or --cc) on macOS/Linux. Timing
builds have no counters. Order rotates across trials; separate instrumented
builds measure proposals. No ablation changes the target or acceptance test.
"""
import argparse
import csv
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--cc', default=os.environ.get('CC', 'clang'))
parser.add_argument('--trials', type=int, default=9)
parser.add_argument('--sizes', default='64,100000')
args = parser.parse_args()
sizes = [int(n) for n in args.sizes.split(',')]
if args.trials < 3 or not sizes or min(sizes) < 1:
    parser.error('need at least three trials and positive batch sizes')
W = args.output.resolve()
W.mkdir(parents=True, exist_ok=True)
current = (ROOT / 'src/c/cb_core.c').read_text()
before = subprocess.check_output(['git', 'show', 'ad17c90:src/c/cb_core.c'], cwd=ROOT, text=True)
variants = {
    'adaptive_before': (before, []),
    'optimized': (current, []),
    'alias_only': (current, ['-DCB_ARS_CDF_MAX=0']),
    'uncached_chords': (current, ['-DCB_ARS_CHORD_CACHE=0']),
    'compiler_inline_choice': (current, ['-DCB_ARS_INLINE=static']),
}
probe = (HERE / 'probe.c').read_text()
libs = {}
for name, (source, flags) in variants.items():
    p = W / (name + '.c')
    p.write_text(source)
    wrapper = W / (name + '_probe.c')
    wrapper.write_text(probe.replace('CORE_FILE', str(p)))
    for counts in [False, True]:
        library = W / (name + ('_counts' if counts else '') + '.so')
        extra = ['-DCB_ARS_DIAGNOSTICS=1'] if counts else ['-DCB_PROFILE_KERNELS=1']
        subprocess.run([args.cc, '-O3', '-shared', '-fPIC', *extra, *flags,
                        '-I', str(ROOT / 'src/c'), str(wrapper), '-lm', '-o', str(library)],
                       check=True, capture_output=True)
        lib = C.CDLL(str(library))
        lib.benchmark.argtypes = [C.c_double, C.c_double, C.c_int, C.c_int, C.c_int,
                                  C.POINTER(C.c_double)]
        if not counts:
            lib.profile_kernels.argtypes = [C.c_double, C.c_double, C.c_int,
                                           C.POINTER(C.c_double)]
        libs[name, counts] = lib

cases = [('tiny7', 7*3.8758505860895066e-32, 7), ('tiny2', 2e-32, 2),
         ('skew5', .5, 5), ('reported594', 594-593.9999812553783, 594),
         ('central10', 4, 10), ('central1000', 400, 1000)]
rows, kernels = [], []
components = ['uniform_rng', 'proposal', 'acceptance_log_uniform', 'squeeze',
              'target_evaluation', 'sigmoid_output']
for label, chi, nu in cases:
    for n in sizes:
        reps = max(10, min(10000, 1000000//n))
        times = {name: [] for name in variants}
        setups = {name: [] for name in variants}
        names = list(variants)
        # Warm every library before interleaving measured trials.
        out = (C.c_double*5)()
        for name in names:
            libs[name, False].benchmark(chi, nu, n, 2, 0, out)
        for trial in range(args.trials):
            order = names[trial % len(names):] + names[:trial % len(names)]
            for name in order:
                libs[name, False].benchmark(chi, nu, n, reps, 0, out)
                assert out[0] > 0, (label, name)
                times[name].append(out[0]*1e9/n)
                libs[name, False].benchmark(chi, nu, n, 1000, 1, out)
                setups[name].append(out[0]*1e6)
        for name in names:
            libs[name, True].benchmark(chi, nu, n, reps, 0, out)
            rows.append(dict(case=label, variant=name, n=n,
                             ns_per_sample=statistics.median(times[name]),
                             min_ns=min(times[name]), max_ns=max(times[name]),
                             setup_us=statistics.median(setups[name]),
                             proposals_per_sample=out[1], insertions_per_batch=out[2],
                             target_evaluations_per_sample=out[3]/n))
    # Weak valid envelopes exercise rescue with identical target parameters.
    for name in ['adaptive_before', 'optimized']:
        out = (C.c_double*5)()
        times = []
        for trial in range(args.trials):
            libs[name, False].benchmark(chi, nu, 100000, 10, 2, out)
            assert out[0] > 0
            times.append(out[0]*1e9/100000)
        libs[name, True].benchmark(chi, nu, 100000, 10, 2, out)
        rows.append(dict(case=label+'_rescue', variant=name, n=100000,
                         ns_per_sample=statistics.median(times), min_ns=min(times),
                         max_ns=max(times), setup_us=0,
                         proposals_per_sample=out[1], insertions_per_batch=out[2],
                         target_evaluations_per_sample=out[3]/100000))
    # These isolated loop costs are deliberately reported separately from
    # complete draws: branch behavior and register allocation differ.
    for name in ['adaptive_before', 'optimized']:
        observations = [[] for _ in components]
        out = (C.c_double*6)()
        for trial in range(args.trials):
            libs[name, False].profile_kernels(chi, nu, 1000000, out)
            for j in range(len(components)):
                observations[j].append(out[j]*1e9)
        for component, values in zip(components, observations):
            kernels.append(dict(case=label, variant=name, component=component,
                                ns_per_operation=statistics.median(values)))
    print(label, 'complete', flush=True)

for filename, data in [('batches', rows), ('kernels', kernels)]:
    with (W / (filename+'.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)
    (W / (filename+'.json')).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
metadata = dict(baseline='ad17c90', source_sha256=hashlib.sha256(current.encode()).hexdigest(),
                platform=platform.platform(), compiler=args.cc+' -O3', trials=args.trials,
                sizes=sizes, timing='Cold exact-ARS hull per batch; constructor, mode solve and core excluded; counter-free timings, separate counters; rotating trial order.')
(W / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
for case, _, _ in cases:
    a = next(r for r in rows if r['case']==case and r['variant']=='adaptive_before' and r['n']==max(sizes))
    b = next(r for r in rows if r['case']==case and r['variant']=='optimized' and r['n']==max(sizes))
    print(f"{case}: {a['ns_per_sample']:.2f} -> {b['ns_per_sample']:.2f} ns/sample ({100*(1-b['ns_per_sample']/a['ns_per_sample']):.1f}% less time)")
