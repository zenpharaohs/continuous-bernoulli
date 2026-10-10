"""Exact endpoint qualification used by the regular validation entry points.

Requires NumPy and a C compiler. Runs in a subprocess with a deadline.
The instrumented build captures eta before sigmoid, so underflow/rounding at
both output endpoints cannot hide a wrong latent distribution. Production
API return/buffer tests use a separate, unmodified build.
"""
from dataclasses import asdict, dataclass
import ctypes as C
import argparse
import csv
import json
import os
import shutil
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np
CORE = pathlib.Path(__file__).resolve().parent / "_csrc"


@dataclass(frozen=True)
class EndpointResult:
    chi: float
    nu: float
    label: str
    path: str
    n: int
    ks: float
    ks_limit: float
    reference_error: float
    passed: bool
    elapsed: float
    kind: str = 'endpoint_eta'

    @property
    def draw_rate(self):
        return self.n / self.elapsed



def bind(path):
    lib = C.CDLL(str(path))
    ptr = C.c_void_p
    doubles = C.POINTER(C.c_double)
    lib.cb_stream_create.argtypes = [C.c_double, C.c_double, C.c_uint64, C.c_int, C.c_int]
    lib.cb_stream_create.restype = ptr
    lib.cb_stream_set_stats.argtypes = [ptr, C.c_double, C.c_double]
    lib.cb_stream_draw.argtypes = [ptr, C.c_int, doubles]
    lib.cb_stream_destroy.argtypes = [ptr]
    lib.cb_stream_peek.argtypes = [ptr, doubles, doubles, C.POINTER(C.c_int), doubles]
    return lib


def run(n=16000):
    results = []
    source = (CORE / 'cb_core.c').read_text()
    compiler = next((shutil.which(c) for c in ('cc', 'clang', 'gcc') if shutil.which(c)), None)
    if compiler is None:
        raise RuntimeError('Exact endpoint validation requires a C compiler (cc, clang, or gcc)')
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for name, text in [('api', source), ('latent', '#define CB_VALIDATE_ETA 1\n' + source + r'''
/* Test only: force exact ARS on either side, retaining the reflected latent
 * coordinate, and exercise the same core/remainder components as the API. */
int latent_draw(double chi, double nu, int n, double *out, int mixture) {
    cb_stream_t *s = cb_stream_create(1, 2, 1729, 0, 64);
    if (mixture == 4) {
        cb_stream_set_stats(s,chi,nu);
        if (!validation_coarse_hull(s)) return 0;
        double old_logZ=s->hull.logZ;
        uint64_t old_insertions=s->ars_insertions;
        if (cb_stream_draw(s,n,out)!=n || s->ars_insertions<=old_insertions ||
            s->hull.m<=2 || s->hull.logZ>old_logZ+4*s->hull.padding) return 0;
        if (s->reflect) for (int i=0;i<n;++i) out[i]=-out[i];
        cb_stream_destroy(s);
        return 5;
    }
    if (mixture == 3) {
        cb_stream_set_stats(s, chi, nu);
        for (int i=0; i<n; ++i)
            if (cb_stream_draw(s,1,out+i) != 1) return 0;
        cb_stream_destroy(s);
        return 4;
    }
    if (mixture == 2) {
        double rate = nu-chi;
        cb_stream_set_stats(s, (nu-1)-.9*rate, nu-1);
        double previous;
        if (cb_stream_draw(s, 1, &previous) != 1) return 0;
        if (cb_stream_draw(s, 1, &previous) != 1) return 0;
        int filled = s->buf_filled, pos = s->buf_pos;
        cb_stream_set_stats(s, chi, nu);
        if (!s->core_valid || s->buf_kind != CB_BUF_CORE_U ||
            s->buf_filled != filled || s->buf_pos != pos || filled <= pos)
            return 0;
        for (int i=0; i<n; ++i)
            if (cb_stream_draw(s,1,out+i) != 1) return 0;
            else out[i] = -out[i]; /* reference uses the lower coordinate */
        cb_stream_destroy(s);
        return 3;
    }
    cb_stream_set_stats(s, chi, nu);
    s->regime = CB_REGIME_ARS;
    s->eta_star = cb_mode(s->chi_eff, nu);
    s->sigma = 1 / sqrt(nu * cb_bft_d2(s->eta_star));
    s->core_valid = 0;
    if (mixture) build_core(s);
    if (s->core_valid) {
        for (int i=0; i<n; ++i) {
            if (cb_rng_uniform(&s->rng) < 1-s->core_alpha_hat) {
                if (!draw_remainder_batch(s, 1, out+i)) return 0;
            } else {
                double u;
                draw_core_batch(s, 1, &u);
                out[i] = s->eta_star + s->sigma*u;
            }
        }
    } else {
        build_hull_exact(s);
        if (!draw_ars_batch(s, n, out)) return 0;
    }
    int status = s->core_valid ? 2 : 1;
    cb_stream_destroy(s);
    return status;
}
int adaptive_reuse(void) {
    /* Invert xorshift64* to force a valid state whose next uniform is zero. */
    cb_rng_t endpoint_rng={11013388057781577488ULL};
    if (cb_rng_uniform(&endpoint_rng)!=0.0) return 0;
    endpoint_rng.s=11013388057781577488ULL;
    double open_u=cb_ars_uniform(&endpoint_rng);
    if (!(open_u>0.0 && open_u<1.0)) return 0;
    cb_stream_t *s=cb_stream_create(1,2,1729,0,64);
    cb_stream_set_stats(s,2e-32,2);
    if (!validation_coarse_hull(s)) return 0;
    uint64_t insertions=s->ars_insertions;
    double x;
    if (cb_stream_draw(s,1,&x)!=1) return 0;
    uint64_t setups=s->ars_setups;
    for (int i=0;i<1024;++i) if (cb_stream_draw(s,1,&x)!=1) return 0;
    if (s->ars_setups!=setups || s->ars_insertions<=insertions) return 0;
    cb_hull_t saved=s->hull;
    if (exact_hull_insert(s,&s->hull,s->hull.pts[0]) || memcmp(&saved,&s->hull,sizeof(saved))) return 0;
    /* Compare the cache to the retained interpolation with identical RNG. */
    cb_stream_t replay=*s;
    double cached[65], interpolated[65];
    for (int i=0;i<s->hull.m;++i) s->hull.chord_slopes[i]=NAN;
    if (!draw_ars_batch(s,65,interpolated) || !draw_ars_batch(&replay,65,cached) ||
        memcmp(cached,interpolated,sizeof(cached))) return 0;
    s->hull=replay.hull;
    cb_stream_set_stats(s,7e-20,7);
    if (cb_stream_draw(s,1,&x)!=1 || s->ars_setups!=setups+1) return 0;
    cb_stream_destroy(s);
    return 1;
}
int bounded_ars_failure(void) {
    const int counts[] = {1,7,65};
    for (int i=0; i<3; ++i) {
        cb_stream_t *s = cb_stream_create(1,2,1729,0,64);
        cb_stream_set_stats(s,2e-32,2);
        /* Test only: invalid numeric scale must propagate through either
         * buffered refill or the batch path without returning a sample. */
        s->eta_star = NAN;
        s->sigma = NAN;
        double out[65];
        int status = cb_stream_draw(s,counts[i],out);
        cb_stream_destroy(s);
        if (status != -1) return 0;
    }
    /* Exercise the finite proposal cap with valid weights but an envelope
     * whose target and chord evaluations are deliberately invalid. */
    cb_stream_t *s = cb_stream_create(1,2,1729,0,64);
    build_hull_exact(s);
    s->chi_eff=NAN;
    for (int i=0;i<s->hull.m;++i) s->hull.fvals[i]=NAN;
    double x;
    int status = draw_ars_batch(s,1,&x);
    cb_stream_destroy(s);
    return status == 0;
}
int bounded_failure(void) {
    cb_stream_t *s = cb_stream_create(594-1.874462168e-5,594,1729,0,64);
    if (!s->core_valid) return 0;
    build_hull_exact(s);
    s->rem_hull=s->hull;
    s->rem_valid=1; s->rem_reflect=s->reflect;
    s->rem_chi_eff=s->chi_eff; s->rem_nu=s->nu;
    s->rem_eta_star=s->eta_star; s->rem_sigma=s->sigma;
    s->rem_log_sub_const=INFINITY;
    /* Force every proposal within core support, making subtraction invalid. */
    s->sigma *= 1000000;
    s->rem_sigma=s->sigma;
    double x;
    s->core_alpha_hat=0; /* Select the invalid remainder with certainty. */
    int ok=cb_stream_draw(s,1,&x);
    cb_stream_destroy(s);
    return ok == -1;
}
''')]:
            cfile = tmp / (name + '.c')
            if os.name == 'nt':
                exports = ['cb_stream_create', 'cb_stream_destroy', 'cb_stream_peek',
                           'cb_stream_set_stats', 'cb_stream_draw']
                if name == 'latent':
                    exports += ['latent_draw', 'bounded_failure', 'bounded_ars_failure', 'adaptive_reuse']
                # Clang targeting MSVC needs explicit exports. MinGW exports
                # these symbols automatically and ignores this guarded block.
                text = '#ifdef _MSC_VER\n' + ''.join(
                    f'#pragma comment(linker, "/export:{symbol}")\n'
                    for symbol in exports) + '#endif\n' + text
            cfile.write_text(text)
            subprocess.run([compiler, '-O2', '-std=c99', '-shared', '-fPIC', str(cfile),
                            '-I', str(CORE), '-o', str(tmp / (name + '.so'))]
                           + ([] if os.name == 'nt' else ['-lm']), check=True,
                           capture_output=True, text=True)
        api = bind(tmp / 'api.so')
        out = np.empty(32)
        op = out.ctypes.data_as(C.POINTER(C.c_double))
        # Minimal replay independent of SPONG or its external trace.
        s = api.cb_stream_create(0, 0, 1729, 0, 64)
        for i in range(1, 3001):
            nu = float(i)
            gap = nu * 3.2e-8
            assert api.cb_stream_set_stats(s, nu-gap, nu) == 1
            sigma = C.c_double()
            api.cb_stream_peek(s, None, None, None, C.byref(sigma))
            actual_gap = nu-(nu-gap)
            assert abs(sigma.value*actual_gap/np.sqrt(nu)-1) < 1e-8
            assert api.cb_stream_draw(s, 1, op) == 1
        # Exact reported states; arbitrary replacements/reflection changes
        # while most of the old buffer is still live.
        states = [(593.9999812553783, 594), (257.99997153659604, 258),
                  (2, 5), (300, 600), (0, 10), (10, 10),
                  (7*3.8758505860895066e-32, 7), (7e-20, 7), (2e-32, 2)]
        states += [(nu-chi, nu) for chi, nu in states[:2]]
        for _ in range(20):
            for chi, nu in states:
                assert api.cb_stream_set_stats(s, chi, nu) == 1
                assert api.cb_stream_draw(s, 7, op) == 7
                assert np.all(np.isfinite(out[:7]))
        # Constructor prefill and requests on either side of buffer capacity.
        for chi, nu in [(7*3.8758505860895066e-32,7), (7e-20,7), (2e-32,2)]:
            fresh = api.cb_stream_create(chi, nu, 1729, 0, 64)
            assert fresh
            batch = np.empty(65)
            for count in (1,7,65):
                assert api.cb_stream_draw(fresh,count,
                    batch.ctypes.data_as(C.POINTER(C.c_double))) == count
                assert np.all(np.isfinite(batch[:count]))
                assert np.all((batch[:count]>0) & (batch[:count]<1))
            api.cb_stream_destroy(fresh)
        trace = pathlib.Path('/tmp/spong_cb_arm0_stats.csv')
        if trace.exists():
            for row in csv.reader(trace.open()):
                assert api.cb_stream_set_stats(s, float(row[1]), float(row[2])) == 1
                assert api.cb_stream_draw(s, 1, op) == 1
            # Optional local trace; the deterministic replacements above are
            # always exercised even on machines without SPONG.
        api.cb_stream_destroy(s)
        latent = bind(tmp / 'latent.so')
        latent.latent_draw.argtypes = [C.c_double, C.c_double, C.c_int,
                                      C.POINTER(C.c_double), C.c_int]
        assert latent.bounded_failure() == 1
        assert latent.bounded_ars_failure() == 1
        assert latent.adaptive_reuse() == 1
        # Independent quadrature of exp(chi*eta-nu*log((exp(eta)-1)/eta))
        # in scaled negative eta. No core, mode solver, or C B function used.
        for gap, nu in [(1.874462168e-5, 594), (2.846340396e-5, 258),
                        (1e-10, 2), (.02, 2), (.01, 49), (2, 200),
                        (7*3.8758505860895066e-32, 7), (7e-20, 7), (2e-32, 2),
                        (7-np.nextafter(7.,0.),7), (2e-150,2), (1,1e6)]:
            for high in [False, True]:
                # No double encodes the reflected interior counterpart when
                # subtraction rounds to nu. Do not test an endpoint in its place.
                if high and nu-gap == nu:
                    continue
                chi = nu-gap if high else gap
                rate = nu-chi if high else chi
                center = -nu/rate
                scale = np.sqrt(nu)/rate
                def reference(size):
                    upper = min(40., max(20., (nu + 45*rate/(nu-rate))/np.sqrt(nu)))
                    grid = np.linspace(-40., upper, size)
                    eta = center + scale*grid
                    b = np.zeros_like(eta)
                    negative = eta < 0
                    positive = eta > 0
                    b[negative] = np.log(-np.expm1(eta[negative]))-np.log(-eta[negative])
                    b[positive] = (eta[positive]+np.log(-np.expm1(-eta[positive]))
                                   -np.log(eta[positive]))
                    logp = rate*eta-nu*b
                    p = np.exp(logp-np.max(logp))
                    masses = .5*(p[:-1]+p[1:])*np.diff(grid)
                    cdf = np.r_[0., np.cumsum(masses)]
                    mass = cdf[-1]
                    # Log concavity bounds each omitted tail by p/|log p'|.
                    edges = eta[[0,-1]]
                    q = np.exp(-np.abs(edges))
                    bp = np.where(edges>0,1/(1-q),-q/(1-q))-1/edges
                    slopes = scale*(rate-nu*bp)
                    assert slopes[0]>0 and slopes[1]<0
                    tail_bound = (p[0]/slopes[0]+p[-1]/(-slopes[1]))/mass
                    assert tail_bound<1e-10
                    cdf /= mass
                    return grid, cdf, tail_bound
                # Independently evaluated density and a convergence check;
                # no C B function, mode solver, or core integral is used.
                coarse_grid, coarse_cdf, coarse_tail = reference(65537)
                grid, cdf, fine_tail = reference(131073)
                reference_error = np.max(np.abs(coarse_cdf-np.interp(coarse_grid, grid, cdf)))+max(coarse_tail,fine_tail)
                if reference_error > 5e-7:
                    raise AssertionError(('endpoint CDF did not converge', reference_error))
                paths = [0, 1, 2] if high and nu in (258, 594) else [0, 1]
                if gap < 1e-18:
                    paths.append(3)
                paths.append(4)
                for mixture in paths:
                    draws = np.empty(n)
                    draw_started = time.monotonic()
                    status = latent.latent_draw(chi, nu, len(draws),
                            draws.ctypes.data_as(C.POINTER(C.c_double)), mixture)
                    assert status in (1, 2, 3, 4, 5), (chi, nu, mixture, status)
                    elapsed = max(time.monotonic()-draw_started, 1e-12)
                    assert np.all(np.isfinite(draws))
                    u = np.sort(np.interp((draws-center)/scale, grid, cdf))
                    idx = np.arange(1, n+1, dtype=float)
                    ks = float(max(np.max(idx/n-u), np.max(u-(idx-1)/n)))
                    path = {1: 'ARS', 2: 'core/remainder', 3: 'live core buffer', 4: 'stream buffer', 5: 'adaptive rescue'}[status]
                    if status == 1 and mixture == 1:
                        path = 'ARS (core unavailable)'
                    label = f"eta gap={rate:.6g} nu={nu} {'high' if high else 'low'} {path}"
                    results.append(EndpointResult(
                        chi=chi, nu=nu, label=label, path=path, n=n, ks=ks,
                        ks_limit=float(3/np.sqrt(n)+reference_error),
                        reference_error=float(reference_error),
                        passed=bool(ks <= 3/np.sqrt(n)+reference_error), elapsed=elapsed))
        if os.name == 'nt':
            # Windows holds loaded DLLs open; release them before deleting the
            # temporary directory. All stream allocations are already freed.
            import _ctypes
            _ctypes.FreeLibrary(api._handle)
            _ctypes.FreeLibrary(latent._handle)
    return results


def run_endpoint_validation(n=16000, timeout=60):
    """Bound the entire compiler, live-buffer, and latent-distribution suite."""
    with tempfile.TemporaryDirectory() as tmp:
        report = pathlib.Path(tmp) / 'endpoint.json'
        completed = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()),
             '--worker', '--n', str(n), '--json-out', str(report)],
            capture_output=True, text=True, timeout=timeout)
        if completed.returncode:
            raise RuntimeError('Exact endpoint validation failed:\n'+completed.stderr)
        rows = json.loads(report.read_text())
    return [EndpointResult(**row) for row in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--n', type=int, default=16000)
    parser.add_argument('--json-out')
    args = parser.parse_args(argv)
    if args.n < 100:
        parser.error('--n must be at least 100')
    results = run(args.n) if args.worker else run_endpoint_validation(args.n)
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps([asdict(r) for r in results]))
    if not args.worker:
        for r in results:
            print(f"{r.label}: KS={r.ks:.5f} limit={r.ks_limit:.5f} {'PASS' if r.passed else 'FAIL'}")
        print('Endpoint live-buffer, mode/scale, bounded-failure checks: PASS')
    # A completed worker always writes its results; the parent includes
    # statistical failures in the normal summary and JSON report.
    return 0 if args.worker or all(r.passed for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
