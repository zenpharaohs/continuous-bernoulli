"""Standalone exact-path regression: python3 tests/exact_endpoint_regression.py.

Requires NumPy/SciPy and a C compiler. Runs in a subprocess with a deadline.
The instrumented build captures eta before sigmoid, so underflow/rounding at
both output endpoints cannot hide a wrong latent distribution. Production
API return/buffer tests use a separate, unmodified build.
"""
import ctypes as C
import csv
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.integrate import quad
from scipy.stats import kstest

ROOT = pathlib.Path(__file__).resolve().parents[1]


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


def run():
    started = time.monotonic()
    source = (ROOT / 'src/c/cb_core.c').read_text()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for name, text in [('api', source), ('latent', source.replace(
                'double th = 1.0 / (1.0 + exp(-ep));', 'double th = ep;').replace(
                'double th = 1.0 / (1.0 + exp(-eta));', 'double th = eta;').replace(
                'if (th <= 0.0) th = 5e-324;', '').replace(
                'if (th >= 1.0) th = nextafter(1.0, 0.0);', '') + r'''
/* Test only: force exact ARS on either side, retaining the reflected latent
 * coordinate, and exercise the same core/remainder components as the API. */
int latent_draw(double chi, double nu, int n, double *out, int mixture) {
    cb_stream_t *s = cb_stream_create(1, 2, 1729, 0, 64);
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
        draw_ars_batch(s, n, out);
    }
    cb_stream_destroy(s);
    return 1;
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
            cfile.write_text(text)
            subprocess.run(['cc', '-O2', '-std=c99', '-shared', '-fPIC', str(cfile),
                            '-I', str(ROOT / 'src/c'), '-o', str(tmp / (name + '.so'))], check=True)
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
                  (2, 5), (300, 600), (0, 10), (10, 10)]
        states += [(nu-chi, nu) for chi, nu in states[:2]]
        for _ in range(20):
            for chi, nu in states:
                assert api.cb_stream_set_stats(s, chi, nu) == 1
                assert api.cb_stream_draw(s, 7, op) == 7
                assert np.all(np.isfinite(out[:7]))
        trace = pathlib.Path('/tmp/spong_cb_arm0_stats.csv')
        if trace.exists():
            for row in csv.reader(trace.open()):
                assert api.cb_stream_set_stats(s, float(row[1]), float(row[2])) == 1
                assert api.cb_stream_draw(s, 1, op) == 1
            print('external replay passed')
        api.cb_stream_destroy(s)
        latent = bind(tmp / 'latent.so')
        latent.latent_draw.argtypes = [C.c_double, C.c_double, C.c_int,
                                      C.POINTER(C.c_double), C.c_int]
        assert latent.bounded_failure() == 1
        # Independent quadrature of exp(chi*eta-nu*log((exp(eta)-1)/eta))
        # in scaled negative eta. No core, mode solver, or C B function used.
        for gap, nu in [(1.874462168e-5, 594), (2.846340396e-5, 258),
                        (1e-10, 2), (.02, 2), (.01, 49), (2, 200)]:
            for high in [False, True]:
                chi = nu-gap if high else gap
                rate = nu-chi if high else chi
                center = -nu/rate
                scale = np.sqrt(nu)/rate
                def logf(u):
                    eta = center + scale*u
                    if eta == 0:
                        b = 0.
                    elif eta < 0:
                        b = np.log(-np.expm1(eta))-np.log(-eta)
                    else:
                        b = eta+np.log(-np.expm1(-eta))-np.log(eta)
                    return rate*eta-nu*b
                peak = logf(0.)
                def density(u):
                    return np.exp(logf(u)-peak)
                upper = max(20., (nu + 45*rate/(nu-rate))/np.sqrt(nu))
                norm = quad(density, -40, upper, epsabs=1e-10, points=[0])[0]
                # Tabulate CDF via quadrature, interpolate for KS.
                grid = np.linspace(-12, min(upper, 12), 2401)
                masses = np.array([quad(density, a, b, epsabs=1e-12)[0]
                                   for a,b in zip(grid[:-1],grid[1:])])
                cdf = np.r_[0., np.cumsum(masses)]/norm
                for mixture in [0, 1]:
                    draws = np.empty(16000)
                    assert latent.latent_draw(chi, nu, len(draws),
                            draws.ctypes.data_as(C.POINTER(C.c_double)), mixture) == 1
                    u = (draws-center)/scale
                    ks = kstest(u, lambda x: np.interp(x, grid, cdf)).statistic
                    assert ks < .018, (gap,nu,high,mixture,ks)
                    print(f'gap={rate:.6g} nu={nu} reflected={high} mixture={mixture} KS={ks:.5f}')
    print(f'all regressions passed in {time.monotonic()-started:.2f}s')


if __name__ == '__main__':
    if '--worker' in sys.argv:
        run()
    else:
        subprocess.run([sys.executable, __file__, '--worker'], check=True, timeout=60)
