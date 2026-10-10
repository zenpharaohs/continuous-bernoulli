"""Reproduce exact ARS costs and an independent 80-digit numerical audit.

Uses only Python's standard library, Git and a C compiler. Timed builds have
no diagnostic counters; proposal counts come from separate instrumented builds.
The historical baseline is fixed at 79bad31 so later commits do not move it.
Reports and temporary shared libraries go to --output. See docs/adaptive-exact-ars.md.
"""
import os
import argparse, ctypes as C, pathlib, subprocess, json, statistics, csv
import decimal as dec, math, bisect
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cc', default=os.environ.get('CC', 'clang'))
parser.add_argument('--quick', action='store_true', help='small sensitivity sweep for regression tests')
parser.add_argument('--output', type=pathlib.Path, required=True)
parser.add_argument('--sensitivity-only', action='store_true')
parser.add_argument('--benchmark-only', action='store_true')
args = parser.parse_args()
if args.sensitivity_only and args.benchmark_only:
    parser.error('choose at most one analysis')
WORK = args.output.resolve()
WORK.mkdir(parents=True, exist_ok=True)
current = (ROOT / 'src/c/cb_core.c').read_text()
baseline = '' if args.sensitivity_only else subprocess.check_output(['git', 'show', '79bad31:src/c/cb_core.c'], cwd=ROOT, text=True)
if not args.sensitivity_only:
    baseline = baseline.replace('uint64_t hull_rebuilds;', '#ifdef CB_ARS_DIAGNOSTICS\n    uint64_t ars_proposals, ars_insertions, ars_evaluations;\n#endif\n    uint64_t hull_rebuilds;')
    a = baseline.index('static int draw_ars_batch')
    b = baseline.index('static void draw_core_batch', a)
    part = baseline[a:b].replace('if (++proposals', '#ifdef CB_ARS_DIAGNOSTICS\n            ++s->ars_proposals;\n#endif\n            if (++proposals').replace('double log_acc = chi*ep', '#ifdef CB_ARS_DIAGNOSTICS\n                ++s->ars_evaluations;\n#endif\n                double log_acc = chi*ep')
    baseline = baseline[:a] + part + baseline[b:]
    (WORK / 'baseline.c').write_text(baseline)
variants = {'baseline': ('baseline.c', ['-DCB_BENCH_BASELINE']), 'shape_fixed': ('current.c', ['-DCB_ARS_SHAPE_INIT=1', '-DCB_ARS_ADAPTIVE=0']), 'shape_adaptive': ('current.c', [])}
if args.sensitivity_only:
    variants = {'shape_adaptive': ('current.c', [])}
(WORK / 'current.c').write_text(current)
for name, (source, flags) in variants.items():
    wrapper = (HERE / 'probe.c').read_text().replace('CORE_FILE', str(WORK / source))
    p = WORK / (name + '_probe.c')
    p.write_text(wrapper)
    for instrumented in [False, True]:
        subprocess.run([args.cc, '-O3', '-shared', '-fPIC', *(['-DCB_ARS_DIAGNOSTICS=1'] if instrumented else []), *flags, '-I', str(ROOT / 'src/c'), str(p), '-lm', '-o', str(WORK / (name + ('_counts' if instrumented else '') + '.so'))], check=True, capture_output=True)

def load(name):
    l = C.CDLL(str(WORK / (name + '.so')))
    l.benchmark.argtypes = [C.c_double, C.c_double, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_double)]
    l.inspect_hull.argtypes = [C.c_double, C.c_double, C.c_int, C.POINTER(C.c_double)]
    return l
if not args.sensitivity_only:
    cases = [('tiny7', 7 * 3.8758505860895066e-32, 7), ('tiny2', 2e-32, 2), ('skew5', 0.5, 5), ('low49', 0.01, 49), ('reported594', 594 - 593.9999812553783, 594), ('central10', 4, 10), ('central1000', 400, 1000)]
    rows = []
    for label, chi, nu in cases:
        for name in variants:
            lib = load(name)
            counts = load(name + '_counts')
            out = (C.c_double * 5)()
            insertion_us = 0.0
            if name == 'shape_adaptive':
                lib.insertion_cost.argtypes = [C.c_double, C.c_double, C.c_int, C.POINTER(C.c_double)]
                lib.insertion_cost(chi, nu, 10000, out)
                insertion_us = out[0] * 1000000.0
                assert out[1] == 10000
            for n in [1, 32, 64, 1024, 100000]:
                lib.benchmark(chi, nu, n, 1000, 1, out)
                setup = out[0]
                reps = max(3, min(1000, 100000 // n))
                times = []
                for _ in range(3):
                    lib.benchmark(chi, nu, n, reps, 0, out)
                    times.append(out[0])
                counts.benchmark(chi, nu, n, reps, 0, out)
                row = dict(insertion_us=insertion_us, case=label, chi=chi, nu=nu, variant=name, n=n, setup_us=setup * 1000000.0, total_us=statistics.median(times) * 1000000.0, proposals_per_sample=out[1], insertions=out[2], evaluations=out[3], hull_points=int(out[4]))
                rows.append(row)
            if name != 'baseline':
                for n in [64, 100000]:
                    lib.benchmark(chi, nu, n, 1000, 3, out)
                    coarse_setup = out[0]
                    reps = max(3, min(1000, 100000 // n))
                    times = []
                    for _ in range(3):
                        lib.benchmark(chi, nu, n, reps, 2, out)
                        times.append(out[0])
                    counts.benchmark(chi, nu, n, reps, 2, out)
                    rows.append(dict(insertion_us=insertion_us, case=label + '_coarse', chi=chi, nu=nu, variant=name, n=n, setup_us=coarse_setup * 1000000.0, total_us=statistics.median(times) * 1000000.0, proposals_per_sample=out[1], insertions=out[2], evaluations=out[3], hull_points=int(out[4])))
        print(label, 'done', flush=True)
    (WORK / 'benchmark.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    with (WORK / 'benchmark.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        if row['n'] in [1, 64, 100000]:
            print(row)
if not args.benchmark_only:
    lib = load('shape_adaptive')

    def hull(chi, nu, refine):
        out = (C.c_double * (4 + 16 * 10))()
        lib.inspect_hull(chi, nu, refine, out)
        m = int(out[0])
        return dict(mode=out[1], sigma=out[2], padding=out[3], points=[list(out[4 + 10 * i:14 + 10 * i]) for i in range(m)])

    def D(x):
        return dec.Decimal.from_float(float(x))

    def b(eta):
        """Independent log partition; omitted small/tail terms are below 1e-80."""
        if not eta:
            return dec.Decimal(0)
        a = abs(eta)
        if a < dec.Decimal('1e-10'):
            return eta / 2 + eta * eta / 24 - eta ** 4 / 2880 + eta ** 6 / 181440
        q = (-a).exp() if a < 200 else dec.Decimal(0)
        return (eta if eta > 0 else dec.Decimal(0)) + (1 - q).ln() - a.ln()

    def bp(eta):
        if not eta:
            return dec.Decimal('.5')
        a = abs(eta)
        if a < dec.Decimal('1e-10'):
            return dec.Decimal('.5') + eta / 12 - eta ** 3 / 720 + eta ** 5 / 30240
        q = (-a).exp() if a < 200 else dec.Decimal(0)
        return (1 / (1 - q) if eta > 0 else -q / (1 - q)) - 1 / eta

    def bpp(eta):
        if not eta:
            return dec.Decimal(1) / 12
        a = abs(eta)
        if a < dec.Decimal('1e-10'):
            return dec.Decimal(1) / 12 - eta ** 2 / 240 + eta ** 4 / 6048
        q = (-a).exp() if a < 200 else dec.Decimal(0)
        return 1 / (eta * eta) - q / ((1 - q) * (1 - q))

    def tangent_max_defect(dc, dn, p, pad):
        """Global maximum of f minus a line, using concavity and an 80-digit mode solve."""
        slope = D(p[2])
        fraction = (dc - slope) / dn
        if not 0 < fraction < 1:
            return dec.Decimal('Infinity')
        eta = D(p[0])
        if fraction < dec.Decimal('1e-6'):
            eta = -1 / fraction
        elif fraction > 1 - dec.Decimal('1e-6'):
            eta = 1 / (1 - fraction)
        else:
            for _ in range(12):
                step = (bp(eta) - fraction) / bpp(eta)
                eta -= step
                if abs(step) < dec.Decimal('1e-60') * (1 + abs(eta)):
                    break
        return dc * eta - dn * b(eta) - (D(p[3]) + slope * eta + pad)
    cases = [(nu * z, nu) for nu in [0.5, 2, 7, 49, 150, 594, 10000.0, 1000000.0] for z in [1e-150, 1e-100, 1e-32, 1e-20, 1e-12, 1e-06, 0.01, 0.1, 0.4, 0.5]]
    cases += [(math.nextafter(nu, 0), nu) for nu in [2, 7, 258, 594, 10000.0, 1000000.0]]
    if args.quick:
        cases = [(5e-151, 0.5), (2e-32, 2), (7e-20, 7), (594 - 593.9999812553783, 594), (1, 1000000.0), (400, 1000), (500000, 1000000.0), (math.nextafter(1000000.0, 0), 1000000.0)]
    rows = []
    perturb = []
    with dec.localcontext() as ctx:
        ctx.prec = 80
        for chi, nu in cases:
            rate = nu - chi if chi > nu / 2 else chi
            dc, dn = (D(rate), D(nu))
            for refined in [0, 1, 2, 3]:
                h = hull(chi, nu, refined)
                pts = h['points']
                mode = h['mode']
                sig = h['sigma']
                pad = D(h['padding'])
                nodes = [p[0] for p in pts]
                zs = [p[4] for p in pts[:-1]]
                assert all((math.isfinite(v) for p in pts for v in p)), (chi, nu, refined)
                grid = [mode + sig * (-12 + 24 * k / 128) for k in range(129)] + nodes + zs
                grid += [math.nextafter(x, direction) for x in nodes + zs for direction in [-math.inf, math.inf]]
                defect = dec.Decimal(0)
                chord_excess = dec.Decimal(0)
                slope_error = dec.Decimal(0)
                f_error = dec.Decimal(0)
                intercept_error = dec.Decimal(0)
                for p in pts:
                    eta = D(p[0])
                    true_f = dc * eta - dn * b(eta)
                    true_s = dc - dn * bp(eta)
                    true_c = true_f - true_s * eta
                    slope_error = max(slope_error, abs(D(p[2]) - true_s) / max(abs(dc), abs(true_s), dec.Decimal('1e-300')))
                    f_error = max(f_error, abs(D(p[1]) - true_f))
                    # Node bounds plus concavity certify every chord, not just the grid.
                    chord_excess = max(chord_excess, D(p[1]) - pad - true_f)
                    intercept_error = max(intercept_error, abs(D(p[3]) - true_c))
                    defect = max(defect, tangent_max_defect(dc, dn, p, pad))
                for x in grid:
                    dx = D(x)
                    true_f = dc * dx - dn * b(dx)
                    j = bisect.bisect_left(zs, x)
                    p = pts[j]
                    env = D(p[3]) + D(p[2]) * dx + pad
                    defect = max(defect, true_f - env)
                    if nodes[0] < x < nodes[-1]:
                        k = max(0, min(len(pts) - 2, bisect.bisect_right(nodes, x) - 1))
                        lower = D(pts[k][1]) + D(pts[k][8])*(dx-D(nodes[k])) - pad
                        chord_excess = max(chord_excess, lower - true_f)
                masses = [0.0] * len(pts)
                for i, p in enumerate(pts):
                    assert 0 <= p[6] <= 1 and 0 <= int(p[7]) < len(pts)
                    masses[i] += p[6] / len(pts)
                    masses[int(p[7])] += (1 - p[6]) / len(pts)
                weights = [math.exp(p[5] - max((p[5] for p in pts))) for p in pts]
                total = sum(weights)
                alias_error = max((abs(a - w / total) for a, w in zip(masses, weights)))
                assert defect == 0 and chord_excess == 0 and (alias_error < 1e-14), (chi, nu, refined, defect, chord_excess, alias_error)
                cdf_masses=[pts[0][9]]+[pts[i][9]-pts[i-1][9] for i in range(1,len(pts))]
                cdf_error=max(abs(a-w/total) for a,w in zip(cdf_masses,weights))
                assert cdf_error<1e-14 and all(a>=0 for a in cdf_masses)
                # Cached chord is a line: endpoint bounds plus concavity
                # certify its whole interval, including after refinement.
                for k in range(len(pts)-1):
                    for endpoint in [nodes[k],nodes[k+1]]:
                        dx=D(endpoint)
                        lower=D(pts[k][1])+D(pts[k][8])*(dx-D(nodes[k]))-pad
                        chord_excess=max(chord_excess,lower-(dc*dx-dn*b(dx)))
                assert chord_excess==0,(chi,nu,refined,chord_excess)
                peak = dc * D(mode) - dn * b(D(mode))
                drop_error = 0
                if not refined & 1:
                    drop_error = max((float(abs(peak - (dc * D(p[0]) - dn * b(D(p[0]))) - D(drop))) for p, drop in zip(pts, [2, 0, 2] if refined & 2 else [5, 2, 0.5, 0, 0.5, 2, 5])))
                rows.append(dict(chi=chi, nu=nu, rate=rate, hull_kind='short' if refined & 2 else 'full', refined=bool(refined & 1), m=len(pts), padding=h['padding'], upper_defect=float(defect), lower_excess=float(chord_excess), slope_relative_error=float(slope_error), density_error=float(f_error), intercept_error=float(intercept_error), drop_error=drop_error, alias_mass_error=alias_error,cdf_mass_error=cdf_error))
            original = hull(chi, nu, 0)
            for name, value, direction in [('chi', chi, -math.inf), ('chi', chi, math.inf), ('nu', nu, -math.inf), ('nu', nu, math.inf)]:
                changed = math.nextafter(value, direction)
                cc = changed if name == 'chi' else chi
                nn = changed if name == 'nu' else nu
                if not 0 < cc < nn:
                    continue
                hp = hull(cc, nn, 0)
                rr = nn - cc if cc > nn / 2 else cc
                perturb.append(dict(chi=chi, nu=nu, coordinate=name, direction='up' if direction > 0 else 'down', rate_relative_change=abs(rr - rate) / rate, mode_shift_sigma=abs(hp['mode'] - original['mode']) / original['sigma'], scale_relative_change=abs(hp['sigma'] / original['sigma'] - 1)))
            print('case', chi, nu, flush=True)
    for name, data in [('sensitivity', rows), ('perturbations', perturb)]:
        with (WORK / (name + '.csv')).open('w') as f:
            w = csv.DictWriter(f, fieldnames=data[0])
            w.writeheader()
            w.writerows(data)
    summary = {key: max(rows, key=lambda r: r[key]) for key in ['upper_defect', 'lower_excess', 'slope_relative_error', 'density_error', 'intercept_error', 'drop_error', 'alias_mass_error','cdf_mass_error']}
    summary['max_perturbation'] = max(perturb, key=lambda r: r['mode_shift_sigma'])
    summary['states'] = len(cases)
    summary['hulls'] = len(rows)
    (WORK / 'sensitivity_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2, allow_nan=False))
