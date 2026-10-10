#!/usr/bin/env python3
"""Profile exact ARS hull placement versus a5f277d; standard Python + Clang.

Timing builds have no counters. Separate diagnostic builds count proposals and
placement density evaluations. Trial order rotates to reduce timing-order bias.
Constructor, mode solve, scale and core construction are outside measured time.
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
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cc', default=os.environ.get('CC', 'clang'))
p.add_argument('--trials', type=int, default=9)
a = p.parse_args()
if a.trials < 3:
    p.error('need at least three trials')
W = a.output.resolve()
W.mkdir(parents=True, exist_ok=True)
current = (ROOT/'src/c/cb_core.c').read_text()
before = subprocess.check_output(['git', 'show', 'a5f277d:src/c/cb_core.c'], cwd=ROOT, text=True)
# Isolate the benefit of finishing Newton before restarting the bounded solve.
start = before.index('static double hull_drop_point')
end = before.index('/* Refresh an exact', start)
placement = before[start:end]
assert placement.count('for (int k=0;k<3;++k)') == 1
newton6 = before[:start] + placement.replace('for (int k=0;k<3;++k)',
                                             'for (int k=0;k<6;++k)') + before[end:]
variants = dict(before=before, newton6=newton6,
                optimized=current)
probe = (HERE/'probe.c').read_text() + '''
void placement(double chi,double nu,int sign,double drop,double*out) {
 cb_stream_t*s=cb_stream_create(0,0,1729,0,64);cb_stream_set_stats(s,chi,nu);
 s->eta_star=cb_mode(s->chi_eff,nu);s->sigma=1/sqrt(nu*cb_bft_d2(s->eta_star));
 double point=hull_drop_point(s,sign,drop);out[0]=point;
#ifdef CB_ARS_DIAGNOSTICS
 out[1]=s->ars_evaluations;
#endif
 out[2]=-target_relative(s,point);cb_stream_destroy(s);
}
'''
libs = {}
for name, source in variants.items():
    core = W/(name+'.c'); core.write_text(source)
    wrapper = W/(name+'_probe.c'); wrapper.write_text(probe.replace('CORE_FILE', str(core)))
    for counts in (False, True):
        so = W/(name+('_counts' if counts else '')+'.so')
        subprocess.run([a.cc, '-O3', '-shared', '-fPIC',
                        *(['-DCB_ARS_DIAGNOSTICS=1'] if counts else []),
                        '-I', str(ROOT/'src/c'), str(wrapper), '-lm', '-o', str(so)],
                       check=True, capture_output=True)
        lib = C.CDLL(str(so))
        lib.benchmark.argtypes = [C.c_double, C.c_double, C.c_int, C.c_int,
                                 C.c_int, C.POINTER(C.c_double)]
        lib.placement.argtypes = [C.c_double, C.c_double, C.c_int, C.c_double,
                                 C.POINTER(C.c_double)]
        libs[name, counts] = lib
cases = [('skew5', .5, 5), ('skew10', .5, 10), ('skew20', 2, 20),
         ('near32', .16, 5), ('central10', 4, 10), ('central1000', 400, 1000),
         ('tiny7', 7e-32, 7), ('reported594', 594-593.9999812553783, 594)]
rows, placements = [], []
names = list(variants)
out = (C.c_double*5)()
for label, chi, nu in cases:
    for n in (1, 64, 100000):
        reps = max(10, min(10000, 1000000//n))
        times = {v: [] for v in names}; setups = {v: [] for v in names}
        for v in names:
            libs[v, False].benchmark(chi, nu, n, 2, 0, out)
        for trial in range(a.trials):
            order = names[trial % len(names):] + names[:trial % len(names)]
            for v in order:
                libs[v, False].benchmark(chi, nu, n, reps, 0, out)
                assert out[0] > 0, (label, v)
                times[v].append(out[0]*1e9/n)
                libs[v, False].benchmark(chi, nu, n, 3000, 1, out)
                setups[v].append(out[0]*1e6)
        for v in names:
            libs[v, True].benchmark(chi, nu, n, reps, 0, out)
            rows.append(dict(case=label, variant=v, n=n,
                             ns_per_sample=statistics.median(times[v]),
                             min_ns=min(times[v]), max_ns=max(times[v]),
                             setup_us=statistics.median(setups[v]),
                             proposals_per_sample=out[1], insertions_per_batch=out[2],
                             evaluations_per_batch=out[3]))
    for v in names:
        for sign in (-1, 1):
            for drop in (.5, 2, 5):
                libs[v, True].placement(chi, nu, sign, drop, out)
                placements.append(dict(case=label, variant=v, sign=sign, drop=drop,
                                       point=out[0], evaluations=out[1], actual_drop=out[2]))
    print(label, 'complete', flush=True)
for filename, data in [('batches', rows), ('placement', placements)]:
    with (W/(filename+'.csv')).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator='\n')
        w.writeheader(); w.writerows(data)
    (W/(filename+'.json')).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
(W/'metadata.json').write_text(json.dumps(dict(baseline='a5f277d',
    source_sha256=hashlib.sha256(current.encode()).hexdigest(), platform=platform.platform(),
    compiler=a.cc+' -O3', trials=a.trials, sizes=[1,64,100000],
    timing='Cold exact ARS hull per batch; excludes constructor, mode, scale and core; rotating trials; counter-free timings and separate diagnostic builds.'), indent=2)+'\n')
for label, _, _ in cases:
    old = next(r for r in rows if r['case']==label and r['variant']=='before' and r['n']==64)
    new = next(r for r in rows if r['case']==label and r['variant']=='optimized' and r['n']==64)
    print(f"{label}: setup {old['setup_us']:.3f} -> {new['setup_us']:.3f} us; cold64 {old['ns_per_sample']:.2f} -> {new['ns_per_sample']:.2f} ns/sample")
