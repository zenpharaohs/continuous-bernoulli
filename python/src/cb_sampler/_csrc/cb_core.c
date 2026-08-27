/*
 * cb_core.c
 * Core C backend for the Continuous Bernoulli conjugate posterior sampler.
 *
 * Three sampling regimes (automatic triage):
 *   GAMMA:  Z_bar < 1/(2*log(nu))  -- tau~Gamma(nu+1,chi), theta=exp(-tau)
 *   CF:     sigma < CB_SIGMA_CF    -- Cornish-Fisher corrected normal
 *   ARS:    otherwise              -- Adaptive Rejection Sampling + squeeze
 *
 * Regime lifetime:
 *   For interior arms (theta_true in [0.20, 0.45]), the Gamma phase is
 *   transient (~3-12 pulls) and the CF phase is late (~300-700 pulls).
 *   ARS dominates the useful pull range and is NOT merely a warm-up.
 *
 * CB_SIGMA_CF = 0.20 (validated in cb_cf_threshold.m):
 *   At sigma < 0.20 the CF sampler passes PIT at N=50000, alpha=0.01.
 *   The previous threshold of 0.25 was too permissive -- sigma in [0.20,0.25]
 *   has detectable approximation error at large N.
 *
 * Compilation:
 *   Standalone test:
 *     gcc -O3 -std=c99 -DCB_CORE_TEST cb_core.c -o cb_core_test -lm
 *   Shared library (Python ctypes):
 *     gcc -O3 -std=c99 -shared -fPIC cb_core.c -o cb_core.so -lm
 *   MEX: see cb_stream_mex.c
 *
 * MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "include/cb_rng.h"
#include "include/cb_bft.h"

/* =========================================================================
 * Configuration
 * ========================================================================= */
#define CB_SIGMA_CF      0.20   /* CF threshold: validated at N=50000 alpha=0.01 */

/* Gamma regime accuracy thresholds.
 * The Gamma sampler draws tau~Gamma(nu+1,chi) and returns theta=exp(-tau).
 * This is the correct large-deviation approximation when Z_bar << 1 (or >>1),
 * but it accumulates detectable bias at N=1M when the posterior width is
 * too large relative to the mode.  Empirically, the KS error scales as
 * ~0.15 * min(Z_bar,1-Z_bar) * sqrt(nu), so both conditions must hold:
 *
 *   (1) min(Z_bar,1-Z_bar) < 0.5/log(nu)        [existing mode-based check]
 *   (2) min(Z_bar,1-Z_bar) * sqrt(nu) < 0.18     [width-of-posterior check]
 *
 * Condition (2) routes to ARS cases like Z=0.10,nu=10 (0.10*3.16=0.316>0.18)
 * while keeping Z=0.05,nu=5 (0.05*2.24=0.112<0.18) and nu=10 (0.158<0.18).
 * Calibration: threshold 0.20 separates all PASS cases (<=0.158) from all
 * FAIL cases (>=0.224) in the N=1M qualification suite (see cb_qualify.m).
 * Using 0.18 adds a 14% safety margin.
 */
#define CB_GAMMA_SQRTN_MAX   0.18   /* max min(Z,1-Z)*sqrt(nu) for Gamma regime */
#define CB_GAMMA_MIN_NU      50.0   /* minimum nu for Gamma regime.
                                    * At small nu, ARS hull rebuild cost is
                                    * amortized over ~nu draws before the arm
                                    * is selected again, so the throughput
                                    * difference vs Gamma is negligible.
                                    * At nu < 50 the approximation error is
                                    * also larger; ARS is correct by construction.
                                    * At nu >= 50 both arguments favor Gamma:
                                    * the arm gets many draws, and the
                                    * tau* = (nu+1)/chi >> 1 condition holds
                                    * well enough that the approximation error
                                    * is undetectable at any practical N. */
#define CB_CF_MIN_NU         50.0   /* minimum nu for CF regime, same rationale. */

/* CF regime skewness threshold.
 * The Cornish-Fisher correction handles the kappa3/sqrt(nu) skewness term
 * explicitly, but leaves a residual KS error ~0.21*|kappa3|/sqrt(nu) due
 * to higher-order terms.  For N=1M the KS critical value is 0.00163, so
 * we require |kappa3|/sqrt(nu) < 0.006 (= 0.00163/0.21 * 0.77 safety).
 * Symmetric posteriors (Z_bar=0.5, kappa3=0) always pass; asymmetric ones
 * with detectable residual skewness are routed to ARS.
 * Calibration from cb_qualify.m N=1M run (see session notes 2026-03-20):
 *   All FAIL CF cases: |kappa3|/sqrt(nu) in [0.013, 0.040] >> 0.006.
 *   All PASS CF cases: kappa3=0 (Z=0.5 by symmetry).
 */
#define CB_CF_SKEW_MAX       0.006  /* max |kappa3|/sqrt(nu) for CF regime */
#define CB_ARS_OVERBOOK  1.3
#define CB_ARS_MIN_BATCH 32
#define CB_ARS_MAX_HULL  9
#define CB_ARS_BASE_HULL 9
#define CB_CORE_K            5
#define CB_CORE_U_MAX        8.0
#define CB_CORE_BUILD_N      257
#define CB_CORE_SHAPE_MARGIN 1e-3
/* Validated W envelope slack.  The deterministic scan currently sees max
 * W excess about 2.5e-4 nats before safety, so 1e-3 is intentional headroom. */
#define CB_CORE_W_SAFETY     1e-3
#define CB_CORE_MIN_ALPHA    0.50
#define CB_CORE_LOG_TAIL     45.0
#define CB_CORE_MIN_NU       150.0
#define CB_CORE_MIN_ROOM     128.0
#define CB_CORE_RANGE_ROOM_FRAC 0.50
#define CB_CORE_RANGE_MAX    256.0
#define CB_CORE_QUAD_N       257
#define CB_CORE_QUAD_WIDTH   12.0
#define CB_CORE_ZIG_K        128
#define CB_DEFAULT_BUF   256

#ifdef CB_CORE_BENCH
static uint64_t bench_core_zig_proposals = 0;
static uint64_t bench_core_zig_squeeze_accepts = 0;
static uint64_t bench_core_zig_shape_tests = 0;
static uint64_t bench_core_zig_shape_accepts = 0;
#endif

/* Adaptive buffer sizing.
 * The ARS hull build cost is amortised over all draws served by that hull.
 * For a dormant suboptimal arm the hull stays valid across many refills;
 * for a rapidly-updated arm it is rebuilt frequently.
 *
 * Policy: track draws_since_rebuild per stream.  Each refill targets
 *   buf_target = min(draws_since_rebuild + CB_DEFAULT_BUF, buf_capacity)
 * so the buffer self-sizes toward buf_capacity for stable arms and
 * resets to CB_DEFAULT_BUF after every hull rebuild.
 *
 * CB_MAX_ADAPTIVE_BUF: upper bound on buf_capacity when the caller
 * supplies buf_size <= 0 ("give me the adaptive default").
 * 4096 doubles = 32 KB -- fits comfortably in L1/L2, one cacheline per
 * typical refill on a stable arm.
 */
#define CB_MAX_ADAPTIVE_BUF  4096

/* Regime codes */
#define CB_REGIME_PRIOR  0
#define CB_REGIME_GAMMA  1
#define CB_REGIME_CF     2
#define CB_REGIME_ARS    3
#define CB_REGIME_POINT  4

/* Buffer payload kind.  ARS transported-core buffers store standardized u;
 * all other buffers store theta samples (possibly sign-bit reflected). */
#define CB_BUF_EMPTY      0
#define CB_BUF_THETA      1
#define CB_BUF_CORE_U     2

/* =========================================================================
 * ARS hull
 * ========================================================================= */
typedef struct {
    int    m;
    double pts   [CB_ARS_MAX_HULL];
    double fvals [CB_ARS_MAX_HULL];
    double slopes[CB_ARS_MAX_HULL];
    double zpts  [CB_ARS_MAX_HULL-1];
    double logwts[CB_ARS_MAX_HULL];
    double logZ;
    /* Precomputed expm1(s*(b-a)) per middle segment (seg 1..m-2).
     * sample_trunc_exp for a middle segment computes:
     *   ep = a + log1p(u * expm1(s*(b-a))) / s   [s > 0]
     *   ep = a + log1p(-u * (-expm1(s*(b-a)))) / s  [s < 0]
     * The expm1(s*(b-a)) term depends only on hull geometry, not on u.
     * Precomputing it saves one expm1 call on every middle-segment proposal.
     * Indices: exm1[i] = expm1(slopes[i] * (zpts[i] - zpts[i-1]))
     * Valid for i = 1..m-2 (middle segments).  i=0 and i=m-1 are tails
     * (semi-infinite intervals) and do not use expm1 in their CDF inversion.
     */
    double exm1 [CB_ARS_MAX_HULL];   /* expm1(s*(b-a)) per segment          */
} cb_hull_t;

/* Transported-core rectangle/alias table over standardized u. */
typedef struct {
    double x_R[CB_CORE_ZIG_K+1], x_L[CB_CORE_ZIG_K+1];
    double y_R[CB_CORE_ZIG_K+1], y_L[CB_CORE_ZIG_K+1];
    double R_cdf[CB_CORE_ZIG_K], L_cdf[CB_CORE_ZIG_K];
    double R_prob[CB_CORE_ZIG_K], L_prob[CB_CORE_ZIG_K];
    int    R_alias[CB_CORE_ZIG_K], L_alias[CB_CORE_ZIG_K];
    double p_right;
    double f0;
} cb_core_zig_t;

/* =========================================================================
 * Stream state
 * ========================================================================= */
typedef struct {
    double chi, nu;
    int    regime;
    int    reflect;     /* 1 if chi/nu > 0.5: sample with chi_eff=nu-chi, return 1-theta.
                        * Exact identity: p(theta|chi,nu) = p(1-theta|nu-chi,nu).
                        * Ensures all samplers operate near theta=0 where doubles
                        * have full resolution.  High-Z precision is documented as
                        * an IEEE 754 limitation; correctness follows by symmetry. */
    double chi_eff;     /* chi if reflect=0, else nu-chi */
    double eta_star, bpp_star, sigma;
    double kappa3, cf_coef;
    int    mode_valid;
    cb_hull_t hull;

    /* Cached exact-current ARS hull. */
    int    hull_valid;       /* 1 = hull current, 0 = needs rebuild       */

    /* Transported common core.  When core_valid=1 and regime=ARS, buf[]
     * stores standardized u-draws from this core.  At drain time, the live
     * posterior transports u via eta = eta_star + sigma*u. */
    int    core_valid;
    int    core_reflect;
    double core_chi0, core_nu0, core_R;
    double core_zA[CB_CORE_K], core_nuA[CB_CORE_K], core_eta[CB_CORE_K], core_sig[CB_CORE_K], core_peak[CB_CORE_K];
    double core_u_grid[CB_CORE_BUILD_N], core_logS_grid[CB_CORE_BUILD_N];
    double core_anchor_grid[CB_CORE_K][CB_CORE_BUILD_N];
    int    core_grid_valid;
    double core_nu1;
    cb_hull_t core_hull;
    double core_logZS;
    double core_alpha_hat;
    cb_core_zig_t core_zig;
    int    core_zig_valid;
    uint64_t core_draws, remainder_draws, core_rebuilds;
    uint64_t rem_cold_builds, rem_cache_hits;

    /* Exact-current remainder sampler cache.  Valid only until the next
     * parameter update or core rebuild; it avoids rebuilding the exact hull
     * and recomputing standardized normalization constants for repeated
     * remainder events at an unchanged parameter. */
    int    rem_valid, rem_reflect;
    double rem_chi_eff, rem_nu, rem_eta_star, rem_sigma;
    double rem_log_sub_const;
    cb_hull_t rem_hull;

    cb_rng_t rng;
    double   normal_spare;
    int      has_spare;
    double  *buf;
    int      buf_capacity, buf_pos, buf_filled, buf_kind;
    uint64_t total_draws, total_refills;
    uint64_t hull_rebuilds;      /* diagnostic: how many times hull was built  */
    uint64_t draws_since_rebuild;/* draws served since last build_hull() call  */
    int      buf_target;         /* adaptive fill size for next do_refill()    */
} cb_stream_t;

/* The sampler uses the closed conjugate family.  (0,0) is the explicit
 * startup prior; for nu>0 the boundary statistics chi=0 and chi=nu are the
 * weak-limit point masses delta_0 and delta_1. */
static int cb_stats_valid(double chi, double nu)
{
    return isfinite(chi) && isfinite(nu)
        && nu >= 0.0 && chi >= 0.0 && chi <= nu;
}

static int cb_observation_valid(double x)
{
    return isfinite(x) && x >= 0.0 && x <= 1.0;
}

static void cb_clip_interior_draws(const cb_stream_t *s, int n, double *out)
{
    if (s->regime == CB_REGIME_POINT) return;
    for (int i = 0; i < n; i++) {
        if (out[i] <= 0.0) out[i] = nextafter(0.0, 1.0);
        if (out[i] >= 1.0) out[i] = nextafter(1.0, 0.0);
    }
}

/* =========================================================================
 * Hull helpers
 * ========================================================================= */
static double log_seg_wt(double fi, double s, double pi, double a, double b)
{
    if (fabs(s) < 1e-8) return fi + log(b - a);
    if (s > 0) {
        double arg = s * (a - b);
        return (arg < -30) ? fi + s*(b-pi) - log(s)
                           : fi + s*(b-pi) - log(s) + log1p(-exp(arg));
    } else {
        double arg = s * (b - a);
        return (arg < -30) ? fi + s*(a-pi) - log(-s)
                           : fi + s*(a-pi) - log(-s) + log1p(-exp(arg));
    }
}

static double logsumexp(const double *v, int n)
{
    double mx = v[0];
    for (int i = 1; i < n; i++) if (v[i] > mx) mx = v[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) s += exp(v[i] - mx);
    return mx + log(s);
}

/* cb_bft_d3_fd() removed: replaced by analytic cb_bft_d3() in cb_bft.h.
 * Derivation: B''=R(eta^2) => B'''=2*eta*R'(eta^2), Horner in eta^2.
 * Accuracy: ~4e-16 vs ~4e-14 for 5-point FD at optimal h. */

static double target_log_f(double chi, double nu, double eta)
{
    return chi * eta - nu * cb_bft(eta);
}

static double cb_mode_warm(double chi, double nu, double eta)
{
    double xbar = chi / nu;
    if (!(xbar > 0.0 && xbar < 1.0) || xbar < 1e-8
            || xbar > 1.0 - 1e-8)
        return cb_mode(chi, nu);
    if (!isfinite(eta))
        eta = log(xbar / (1.0 - xbar));

    int converged = 0;
    for (int iter = 0; iter < 12; iter++) {
        double bp   = cb_bft_d1(eta);
        double bpp  = cb_bft_d2(eta);
        if (!isfinite(bp) || !isfinite(bpp) || !(bpp > 0.0))
            return cb_mode(chi, nu);
        double step = (bp - xbar) / bpp;
        if (!isfinite(step))
            return cb_mode(chi, nu);
        eta -= step;
        if (fabs(step) < 1e-14 * (1.0 + fabs(eta))) {
            converged = 1;
            break;
        }
    }
    if (!converged || !isfinite(eta))
        return cb_mode(chi, nu);
    if (fabs(cb_bft_d1(eta) - xbar) > 1e-12)
        return cb_mode(chi, nu);
    return eta;
}

static double logdiffexp(double a, double b)
{
    if (b >= a) return -INFINITY;
    return a + log1p(-exp(b - a));
}

static double sample_trunc_exp(cb_rng_t *r, double s, double a, double b);

static void hull_offsets(double sigma, int *m_out, double *offsets)
{
    if (sigma <= 1.5) {
        double o[] = {-3.0, -1.5, 0.0, 1.5, 3.0};
        *m_out = 5;
        memcpy(offsets, o, 5*sizeof(double));
    } else if (sigma <= 3.0) {
        double o[] = {-3.5, -2.25, -1.0, 0.0, 1.0, 2.25, 3.5};
        *m_out = 7;
        memcpy(offsets, o, 7*sizeof(double));
    } else {
        double o[] = {-3.5,-2.5,-1.75,-0.875,0.0,0.875,1.75,2.5,3.5};
        *m_out = 9;
        memcpy(offsets, o, 9*sizeof(double));
    }
}

static void build_hull_exact(cb_stream_t *s)
{
    double chi = s->chi_eff, nu = s->nu;  /* chi_eff: low-Z always */
    double eta_star = s->eta_star, sigma = s->sigma;

    int m;
    double offsets[CB_ARS_BASE_HULL];
    hull_offsets(sigma, &m, offsets);

    cb_hull_t *h = &s->hull;
    h->m = m;
    for (int i = 0; i < m; i++) {
        h->pts[i]    = eta_star + sigma * offsets[i];
        h->fvals[i]  = chi * h->pts[i] - nu * cb_bft(h->pts[i]);
        h->slopes[i] = chi - nu * cb_bft_d1(h->pts[i]);
    }
    for (int i = 0; i < m-1; i++) {
        double ds = h->slopes[i] - h->slopes[i+1];
        h->zpts[i] = (fabs(ds) < 1e-14)
            ? 0.5*(h->pts[i] + h->pts[i+1])
            : (h->fvals[i+1] - h->fvals[i]
               + h->slopes[i]*h->pts[i] - h->slopes[i+1]*h->pts[i+1]) / ds;
    }
    /* Precompute exm1[i] = expm1(slopes[i] * (zpts[i] - zpts[i-1]))
     * for middle segments i = 1..m-2.  Used in draw_ars_batch to avoid
     * recomputing expm1(s*(b-a)) on every proposal from that segment.
     * Tail segments (i=0, i=m-1) are semi-infinite; their CDF inversions
     * use log(u) and log1p(-u) respectively -- no expm1 involved. */
    for (int i = 1; i < m-1; i++)
        h->exm1[i] = expm1(h->slopes[i] * (h->zpts[i] - h->zpts[i-1]));

    h->logwts[0]   = h->fvals[0]   + h->slopes[0]  *(h->zpts[0]   - h->pts[0])
                     - log( h->slopes[0]);
    h->logwts[m-1] = h->fvals[m-1] + h->slopes[m-1]*(h->zpts[m-2] - h->pts[m-1])
                     - log(-h->slopes[m-1]);
    for (int i = 1; i < m-1; i++)
        h->logwts[i] = log_seg_wt(h->fvals[i], h->slopes[i], h->pts[i],
                                   h->zpts[i-1], h->zpts[i]);
    h->logZ = logsumexp(h->logwts, m);

    /* Reset draw counter for adaptive sizing. */
    s->hull_valid          = 1;
    s->hull_rebuilds++;
    s->draws_since_rebuild = 0;
    s->buf_target          = CB_DEFAULT_BUF;  /* restart conservative after rebuild */
}

static void build_hull(cb_stream_t *s)
{
    build_hull_exact(s);
}

static void copy_hull(cb_hull_t *dst, const cb_hull_t *src)
{
    memcpy(dst, src, sizeof(cb_hull_t));
}

static int binom_wait2_rem(cb_rng_t *rng, int n, double p)
{
    if (p <= 0.0) return 0;
    if (p >= 1.0) return n;

    int flip = 0;
    if (p > 0.5) { p = 1.0 - p; flip = 1; }
    double inv_lam = 1.0 / (-log1p(-p));
    int t = 0, k = 0;
    while (t < n) {
        double u = cb_rng_uniform(rng);
        if (u <= 0.0) u = 5e-324;
        int G = (int)(-log(u) * inv_lam);
        t += G + 1;
        if (t <= n) k++;
    }
    return flip ? (n - k) : k;
}

static void hull_proposal_pre(cb_stream_t *s, const double *probs, double sum,
                              double *eta_out, double *log_env_out)
{
    const cb_hull_t *h = &s->hull;
    int m = h->m;
    double u = cb_rng_uniform(&s->rng) * sum;
    double cdf = 0.0;
    int seg = m - 1;
    for (int i = 0; i < m-1; i++) {
        cdf += probs[i];
        if (u <= cdf) { seg = i; break; }
    }

    double sl = h->slopes[seg];
    double fv = h->fvals[seg];
    double pp = h->pts[seg];
    double a  = (seg == 0)   ? -INFINITY : h->zpts[seg-1];
    double b  = (seg == m-1) ?  INFINITY : h->zpts[seg];
    double u2 = cb_rng_uniform(&s->rng);
    double ep;
    if (seg == 0) ep = b + log(u2) / sl;
    else if (seg == m-1) ep = a + log1p(-u2) / sl;
    else if (fabs(sl) < 1e-8) ep = a + u2 * (b - a);
    else if (sl > 0.0) ep = a + log1p( u2 *  h->exm1[seg]) / sl;
    else ep = a + log1p(-u2 * (-h->exm1[seg])) / sl;

    *eta_out = ep;
    *log_env_out = fv + sl * (ep - pp);
}

static double core_anchor_log_shape(const cb_stream_t *s, int a, double u)
{
    double chi = s->core_zA[a] * s->core_nuA[a];
    double eta = s->core_eta[a] + s->core_sig[a] * u;
    return target_log_f(chi, s->core_nuA[a], eta) - s->core_peak[a];
}

static int core_active_anchor(const cb_stream_t *s, double u, double *val_out)
{
    int best = 0;
    double bestv = core_anchor_log_shape(s, 0, u);
    for (int a = 1; a < CB_CORE_K; a++) {
        double v = core_anchor_log_shape(s, a, u);
        if (v < bestv) {
            bestv = v;
            best = a;
        }
    }
    if (val_out) *val_out = bestv - CB_CORE_SHAPE_MARGIN;
    return best;
}

static double core_log_shape(const cb_stream_t *s, double u)
{
    if (u < -CB_CORE_U_MAX || u > CB_CORE_U_MAX) return -INFINITY;
    double v;
    core_active_anchor(s, u, &v);
    return v;
}

static void core_build_grid(cb_stream_t *s)
{
    double h = (2.0 * CB_CORE_U_MAX) / (double)(CB_CORE_BUILD_N - 1);
    for (int i = 0; i < CB_CORE_BUILD_N; i++) {
        double u = -CB_CORE_U_MAX + h * (double)i;
        s->core_u_grid[i] = u;

        double bestv = INFINITY;
        for (int a = 0; a < CB_CORE_K; a++) {
            double v = core_anchor_log_shape(s, a, u);
            s->core_anchor_grid[a][i] = v;
            if (v < bestv) bestv = v;
        }
        s->core_logS_grid[i] = bestv - CB_CORE_SHAPE_MARGIN;
    }
    s->core_grid_valid = 1;
}

static double core_shape_slope(const cb_stream_t *s, double u)
{
    double v;
    int a = core_active_anchor(s, u, &v);
    double chi = s->core_zA[a] * s->core_nuA[a];
    double eta = s->core_eta[a] + s->core_sig[a] * u;
    return s->core_sig[a] * (chi - s->core_nuA[a] * cb_bft_d1(eta));
}

static double core_logW_anchor(const cb_stream_t *s, int a)
{
    double logs[CB_CORE_BUILD_N];
    double h = (2.0 * CB_CORE_U_MAX) / (double)(CB_CORE_BUILD_N - 1);
    for (int i = 0; i < CB_CORE_BUILD_N; i++) {
        double w = (i == 0 || i == CB_CORE_BUILD_N - 1) ? 1.0 : ((i & 1) ? 4.0 : 2.0);
        double v = s->core_grid_valid
                 ? s->core_anchor_grid[a][i]
                 : core_anchor_log_shape(s, a, -CB_CORE_U_MAX + h * (double)i);
        logs[i] = v + log(w);
    }
    return log(h / 3.0) + logsumexp(logs, CB_CORE_BUILD_N);
}

static int core_certify_candidate(const cb_stream_t *s, double chi, double nu)
{
    double eta = cb_mode(chi, nu);
    double sig = 1.0 / sqrt(nu * cb_bft_d2(eta));
    double peak = target_log_f(chi, nu, eta);
    double h = (2.0 * CB_CORE_U_MAX) / (double)(CB_CORE_BUILD_N - 1);
    for (int i = 0; i < CB_CORE_BUILD_N; i++) {
        double u = s->core_grid_valid ? s->core_u_grid[i]
                                      : -CB_CORE_U_MAX + h * (double)i;
        double g = target_log_f(chi, nu, eta + sig * u) - peak;
        double logS = s->core_grid_valid ? s->core_logS_grid[i]
                                         : core_log_shape(s, u);
        if (logS > g + 1e-12)
            return 0;
    }
    return 1;
}

static int core_certify_range(const cb_stream_t *s)
{
    double chi0 = s->core_chi0, nu0 = s->core_nu0, R = s->core_R;
    static const double fracs[] = {0.0, 0.25, 0.5, 0.75, 1.0};
    static const double edges[] = {0.0, 0.5, 1.0};
    for (int i = 0; i < 5; i++) {
        double dnu = R * fracs[i];
        for (int j = 0; j < 3; j++) {
            double dchi = dnu * edges[j];
            if (!core_certify_candidate(s, chi0 + dchi, nu0 + dnu))
                return 0;
        }
    }
    for (int a = 0; a < CB_CORE_K; a++) {
        double t = (CB_CORE_K == 1) ? 0.0 : (double)a / (double)(CB_CORE_K - 1);
        if (!core_certify_candidate(s, chi0 + R * t, nu0 + R))
            return 0;
    }
    return 1;
}

static double core_logZS_integral(const cb_stream_t *s)
{
    double logs[CB_CORE_BUILD_N];
    double h = (2.0 * CB_CORE_U_MAX) / (double)(CB_CORE_BUILD_N - 1);
    for (int i = 0; i < CB_CORE_BUILD_N; i++) {
        double w = (i == 0 || i == CB_CORE_BUILD_N - 1) ? 1.0 : ((i & 1) ? 4.0 : 2.0);
        double v = s->core_grid_valid
                 ? s->core_logS_grid[i]
                 : core_log_shape(s, -CB_CORE_U_MAX + h * (double)i);
        logs[i] = v + log(w);
    }
    return log(h / 3.0) + logsumexp(logs, CB_CORE_BUILD_N);
}

static void core_zig_make_alias(const double *weights, double total,
                                double *prob, int *alias)
{
    double scaled[CB_CORE_ZIG_K];
    int small[CB_CORE_ZIG_K], large[CB_CORE_ZIG_K];
    int ns = 0, nl = 0;

    for (int i = 0; i < CB_CORE_ZIG_K; i++) {
        scaled[i] = weights[i] * (double)CB_CORE_ZIG_K / total;
        if (scaled[i] < 1.0) small[ns++] = i;
        else large[nl++] = i;
    }

    while (ns > 0 && nl > 0) {
        int sidx = small[--ns];
        int lidx = large[--nl];
        prob[sidx] = scaled[sidx];
        alias[sidx] = lidx;
        scaled[lidx] = (scaled[lidx] + scaled[sidx]) - 1.0;
        if (scaled[lidx] < 1.0) small[ns++] = lidx;
        else large[nl++] = lidx;
    }

    while (nl > 0) {
        int i = large[--nl];
        prob[i] = 1.0;
        alias[i] = i;
    }
    while (ns > 0) {
        int i = small[--ns];
        prob[i] = 1.0;
        alias[i] = i;
    }
}

static int core_hull_build(cb_stream_t *s)
{
    static const double pts[5] = {-3.0, -1.5, 0.0, 1.5, 3.0};
    cb_hull_t *h = &s->core_hull;
    h->m = 5;
    for (int i = 0; i < h->m; i++) {
        h->pts[i] = pts[i];
        h->fvals[i] = core_log_shape(s, pts[i]);
        h->slopes[i] = core_shape_slope(s, pts[i]);
    }
    for (int i = 0; i < h->m - 1; i++) {
        double ds = h->slopes[i] - h->slopes[i+1];
        h->zpts[i] = (fabs(ds) < 1e-14)
            ? 0.5 * (h->pts[i] + h->pts[i+1])
            : (h->fvals[i+1] - h->fvals[i]
               + h->slopes[i] * h->pts[i] - h->slopes[i+1] * h->pts[i+1]) / ds;
        if (h->zpts[i] <= -CB_CORE_U_MAX || h->zpts[i] >= CB_CORE_U_MAX)
            return 0;
        if (i > 0 && h->zpts[i] <= h->zpts[i-1])
            return 0;
    }
    for (int i = 1; i < h->m - 1; i++)
        h->exm1[i] = expm1(h->slopes[i] * (h->zpts[i] - h->zpts[i-1]));
    h->logwts[0] = log_seg_wt(h->fvals[0], h->slopes[0], h->pts[0],
                               -CB_CORE_U_MAX, h->zpts[0]);
    h->logwts[h->m-1] = log_seg_wt(h->fvals[h->m-1], h->slopes[h->m-1], h->pts[h->m-1],
                                    h->zpts[h->m-2], CB_CORE_U_MAX);
    for (int i = 1; i < h->m - 1; i++)
        h->logwts[i] = log_seg_wt(h->fvals[i], h->slopes[i], h->pts[i],
                                   h->zpts[i-1], h->zpts[i]);
    h->logZ = logsumexp(h->logwts, h->m);
    return 1;
}

static int core_zig_build(cb_stream_t *s)
{
    cb_core_zig_t *z = &s->core_zig;
    int K = CB_CORE_ZIG_K;
    double area_R[CB_CORE_ZIG_K], area_L[CB_CORE_ZIG_K];
    z->f0 = core_log_shape(s, 0.0);
    if (!isfinite(z->f0)) return 0;

    double sum_R = 0.0, sum_L = 0.0;
    for (int k = 0; k <= K; k++) {
        double x = CB_CORE_U_MAX * (double)k / (double)K;
        z->x_R[k] = x;
        z->x_L[k] = x;
        z->y_R[k] = exp(core_log_shape(s,  x) - z->f0);
        z->y_L[k] = exp(core_log_shape(s, -x) - z->f0);
        if (!isfinite(z->y_R[k]) || !isfinite(z->y_L[k])) return 0;
        if (k > 0 && (z->y_R[k] > z->y_R[k-1] + 1e-12 ||
                      z->y_L[k] > z->y_L[k-1] + 1e-12))
            return 0;
    }

    for (int k = 0; k < K; k++) {
        double aR = (z->x_R[k+1] - z->x_R[k]) * z->y_R[k];
        double aL = (z->x_L[k+1] - z->x_L[k]) * z->y_L[k];
        area_R[k] = aR;
        area_L[k] = aL;
        sum_R += aR;
        sum_L += aL;
        z->R_cdf[k] = sum_R;
        z->L_cdf[k] = sum_L;
    }
    if (!(sum_R > 0.0) || !(sum_L > 0.0)) return 0;

    for (int k = 0; k < K; k++) {
        z->R_cdf[k] /= sum_R;
        z->L_cdf[k] /= sum_L;
    }
    z->R_cdf[K-1] = 1.0;
    z->L_cdf[K-1] = 1.0;
    core_zig_make_alias(area_R, sum_R, z->R_prob, z->R_alias);
    core_zig_make_alias(area_L, sum_L, z->L_prob, z->L_alias);
    z->p_right = sum_R / (sum_R + sum_L);
    return isfinite(z->p_right) && z->p_right > 0.0 && z->p_right < 1.0;
}

static int core_covers_current(const cb_stream_t *s)
{
    if (!s->core_valid) return 0;
    if (s->reflect != s->core_reflect) return 0;
    double dnu = s->nu - s->core_nu0;
    double dchi = s->chi_eff - s->core_chi0;
    double eps = 1e-10;
    return dnu >= -eps
        && dnu <= s->core_R + eps
        && dchi >= -eps
        && dchi <= dnu + eps;
}

static int core_prepare_candidate(cb_stream_t *s, double chi0, double nu0,
                                  double R, double *alpha_out)
{
    s->core_R = R;
    s->core_nu1 = nu0 + R;

    s->core_zA[0] = chi0 / nu0;
    s->core_nuA[0] = nu0;
    s->core_eta[0] = cb_mode(chi0, nu0);
    s->core_sig[0] = 1.0 / sqrt(nu0 * cb_bft_d2(s->core_eta[0]));
    s->core_peak[0] = target_log_f(chi0, nu0, s->core_eta[0]);

    double zlo = chi0 / s->core_nu1;
    double zhi = (chi0 + R) / s->core_nu1;
    for (int a = 0; a < CB_CORE_K; a++) {
        if (a == 0) continue;
        double t = (CB_CORE_K == 2) ? 0.0 : (double)(a - 1) / (double)(CB_CORE_K - 2);
        double z = zlo + t * (zhi - zlo);
        double chi = z * s->core_nu1;
        double eta = cb_mode(chi, s->core_nu1);
        s->core_zA[a] = z;
        s->core_nuA[a] = s->core_nu1;
        s->core_eta[a] = eta;
        s->core_sig[a] = 1.0 / sqrt(s->core_nu1 * cb_bft_d2(eta));
        s->core_peak[a] = target_log_f(chi, s->core_nu1, eta);
    }

    core_build_grid(s);
    if (!core_certify_range(s)) return 0;

    s->core_logZS = core_logZS_integral(s);
    double logWmax = core_logW_anchor(s, 0);
    for (int a = 1; a < CB_CORE_K; a++) {
        double lw = core_logW_anchor(s, a);
        if (lw > logWmax) logWmax = lw;
    }
    double alpha = exp(s->core_logZS - logWmax - CB_CORE_W_SAFETY);
    if (!(alpha >= CB_CORE_MIN_ALPHA)) return 0;
    if (alpha > 1.0) alpha = 1.0;
    if (alpha_out) *alpha_out = alpha;
    return 1;
}

static void build_core(cb_stream_t *s)
{
    s->core_valid = 0;
    s->core_grid_valid = 0;
    s->core_zig_valid = 0;
    s->rem_valid = 0;
    double chi0 = s->chi_eff, nu0 = s->nu;
    double room = nu0 - 2.0 * chi0;
    if (nu0 < CB_CORE_MIN_NU || room < CB_CORE_MIN_ROOM) return;

    s->core_reflect = s->reflect;
    s->core_chi0 = chi0;
    s->core_nu0 = nu0;

    double R = CB_CORE_RANGE_ROOM_FRAC * room;
    if (R > CB_CORE_RANGE_MAX) R = CB_CORE_RANGE_MAX;
    if (R < CB_CORE_MIN_ROOM) return;

    double alpha = 0.0, default_alpha = 0.0;
    if (!core_prepare_candidate(s, chi0, nu0, R, &default_alpha)) return;
    alpha = default_alpha;

    s->core_alpha_hat = alpha;
    if (!core_hull_build(s)) return;
    s->core_zig_valid = core_zig_build(s);

    s->core_valid = 1;
    s->hull_rebuilds++;
    s->core_rebuilds++;
    s->draws_since_rebuild = 0;
    s->buf_target = CB_DEFAULT_BUF;
}

static void sample_core_proposal(cb_stream_t *s, double *u_out, double *log_env_out)
{
    const cb_hull_t *h = &s->core_hull;
    double probs[CB_ARS_MAX_HULL];
    double mx = h->logwts[0], sum = 0.0;
    for (int i = 1; i < h->m; i++) if (h->logwts[i] > mx) mx = h->logwts[i];
    for (int i = 0; i < h->m; i++) { probs[i] = exp(h->logwts[i] - mx); sum += probs[i]; }

    double u = cb_rng_uniform(&s->rng) * sum;
    double cdf = 0.0;
    int seg = h->m - 1;
    for (int i = 0; i < h->m - 1; i++) {
        cdf += probs[i];
        if (u <= cdf) { seg = i; break; }
    }

    double a = (seg == 0) ? -CB_CORE_U_MAX : h->zpts[seg-1];
    double b = (seg == h->m - 1) ? CB_CORE_U_MAX : h->zpts[seg];
    double sl = h->slopes[seg];
    double ep = sample_trunc_exp(&s->rng, sl, a, b);
    *u_out = ep;
    *log_env_out = h->fvals[seg] + sl * (ep - h->pts[seg]);
}

static double current_logW_standardized(cb_stream_t *s)
{
    double logs[CB_CORE_QUAD_N];
    double h = (2.0 * CB_CORE_QUAD_WIDTH) / (double)(CB_CORE_QUAD_N - 1);
    double peak = target_log_f(s->chi_eff, s->nu, s->eta_star);
    for (int i = 0; i < CB_CORE_QUAD_N; i++) {
        double u = -CB_CORE_QUAD_WIDTH + h * (double)i;
        double eta = s->eta_star + s->sigma * u;
        double w = (i == 0 || i == CB_CORE_QUAD_N - 1) ? 1.0 : ((i & 1) ? 4.0 : 2.0);
        logs[i] = target_log_f(s->chi_eff, s->nu, eta) - peak + log(w);
    }
    return log(h / 3.0) + logsumexp(logs, CB_CORE_QUAD_N);
}

/* =========================================================================
 * Regime determination
 * ========================================================================= */
static void determine_regime(cb_stream_t *s)
{
    double chi = s->chi, nu = s->nu;
    int old_reflect = s->reflect;

    if (nu <= 0.0) {
        s->regime   = CB_REGIME_PRIOR;
        s->reflect  = 0;
        s->chi_eff  = chi;
        s->sigma    = 1e10;
        s->hull_valid = 0;
        s->core_valid = 0;
        s->mode_valid = 0;
        return;
    }

    /* Closed-family boundary.  The density on finite eta is improper at
     * exactly chi=0 or chi=nu, but the proper family has the weak limits
     * delta_0 and delta_1.  Represent those limits directly: never give a
     * boundary target to a rejection sampler. */
    if (chi == 0.0 || chi == nu) {
        s->regime = CB_REGIME_POINT;
        s->reflect = (chi == nu);
        s->chi_eff = 0.0;
        s->eta_star = s->reflect ? INFINITY : -INFINITY;
        s->bpp_star = 0.0;
        s->sigma = 0.0;
        s->hull_valid = 0;
        s->core_valid = 0;
        s->mode_valid = 0;
        return;
    }

    /* Reflection: always use the low-Z parameterisation for ARS.
     * For chi/nu > 0.5 we sample from (nu-chi, nu) and return 1-theta.
     * This keeps the hull operating near theta=0 where doubles
     * have full resolution.  Gamma and CF already handle reflection
     * internally (draw_gamma_batch uses chi_eff directly; CF works in
     * eta-space which is symmetric).  For ARS we store chi_eff here and
     * use it in build_hull and the output conversion. */
    s->reflect = (chi / nu > 0.5);
    s->chi_eff = s->reflect ? (nu - chi) : chi;

    double zbar    = chi / nu;
    double log_nu  = (nu > 2.0) ? log(nu) : log(2.0);
    double gam_thr = 0.5 / log_nu;

    /* Gamma regime: Z_bar near 0 or 1, AND posterior width is small enough
     * that the tau~Gamma(nu+1,chi) approximation is accurate.
     * Condition (2) min(Z_bar,1-Z_bar)*sqrt(nu) < CB_GAMMA_SQRTN_MAX tightens
     * the original threshold for cases where nu is moderate and Z_bar is not
     * sufficiently extreme (e.g. Z=0.10,nu=10 has 0.10*3.16=0.316 > 0.18). */
    {
        double zmin    = (zbar < 0.5) ? zbar : 1.0 - zbar;
        double width_ok = zmin * sqrt(nu) < CB_GAMMA_SQRTN_MAX;
        if (nu >= CB_GAMMA_MIN_NU && (zbar < gam_thr || (1.0 - zbar) < gam_thr) && width_ok) {
            /* Precision guard for the high-Z reflected branch.
             * phi = exp(-tau) rounds to 0 (i.e. 1-phi rounds to 1.0) when
             * tau > -log(eps/2) ~ 36.7.  For Gamma(nu+1, rate) with
             * rate = nu-chi, P(tau > 36.7) = 1 - GammaCDF(36.7*rate, nu+1).
             * When mean_tau = (nu+1)/rate > 36.7/10 = 3.67, this probability
             * exceeds ~0.01% and creates a detectable point mass at theta=1.
             * Route such cases to ARS, which works in eta-space.
             * Low-Z branch (rate=chi) is immune: phi=exp(-tau) representable
             * to tau~745, so no precision guard needed there.
             */
            if (zbar > 0.5) {
                double rate_hi = nu - chi;
                double mean_tau_hi = (nu + 1.0) / rate_hi;
                if (mean_tau_hi > 3.67) goto ars_regime;
            }
            s->regime = CB_REGIME_GAMMA;
            s->sigma  = 0.0;
            s->hull_valid = 0;
            s->mode_valid = 0;
            return;
        }
    }

    /* Use chi_eff for mode/sigma so hulls are built for the
     * low-Z reflected distribution when reflect=1. */
    double chi_e    = s->chi_eff;
    double eta_star = (s->mode_valid && old_reflect == s->reflect)
                    ? cb_mode_warm(chi_e, nu, s->eta_star)
                    : cb_mode(chi_e, nu);
    double bpp      = cb_bft_d2(eta_star);
    double sigma    = 1.0 / sqrt(nu * bpp);

    s->eta_star = eta_star;
    s->bpp_star = bpp;
    s->sigma    = sigma;
    s->mode_valid = 1;

    if (nu >= CB_CF_MIN_NU && sigma < CB_SIGMA_CF) {
        /* CF regime: also require the standardized skewness to be small.
         * The residual KS error scales as ~0.21*|kappa3|/sqrt(nu), so
         * cases with |kappa3|/sqrt(nu) >= CB_CF_SKEW_MAX are routed to ARS.
         * This routes all asymmetric CF cases (Z != 0.5) where the CF
         * correction leaves a detectable systematic error at N=1M. */
        double bppp = cb_bft_d3(eta_star);
        double k3   = bppp / pow(bpp, 1.5);
        if (fabs(k3) / sqrt(nu) >= CB_CF_SKEW_MAX) goto ars_regime;
        s->regime  = CB_REGIME_CF;
        s->kappa3  = k3;
        s->cf_coef = k3 / (6.0 * sqrt(nu));
        s->hull_valid = 0;   /* CF doesn't use the hull */
    } else {
        ars_regime:
        s->regime = CB_REGIME_ARS;
        if (!core_covers_current(s)) {
            s->core_valid = 0;
        }
        s->hull_valid = 0;  /* exact hull is current-parameter only */
    }
}

/* =========================================================================
 * Truncated exponential sampler
 * ========================================================================= */
static double sample_trunc_exp(cb_rng_t *r, double s, double a, double b)
{
    double u = cb_rng_uniform(r);
    if (fabs(s) < 1e-8)        return a + u*(b - a);
    if (s > 0 && a <= -INFINITY) return b + log(u) / s;
    if (s < 0 && b >=  INFINITY)  return a + log1p(-u) / s;
    if (s > 0)                    return a + log1p(u * expm1(s*(b - a))) / s;
    return                               a + log1p(-u * (-expm1(s*(b - a)))) / s;
}

/* =========================================================================
 * Batch draw routines
 * ========================================================================= */
static void draw_gamma_batch(cb_stream_t *s, int n, double *out)
{
    double chi = s->chi, nu = s->nu, shape = nu + 1.0;
    /* Always use the low-Z parameterisation via the exact reflection:
     *   p(theta | chi, nu) = p(1-theta | nu-chi, nu)
     * which follows from B(eta) - B(-eta) = eta.
     *
     * Low-Z path:   rate = chi,     phi = exp(-tau), theta = phi
     * High-Z path:  rate = nu-chi,  phi = exp(-tau), theta = 1 - phi
     *
     * phi = exp(-tau) is always a positive representable double.
     * For high-Z, 1-phi may round to 1.0 when tau > 36.7 (phi < eps/2).
     * The determine_regime() precision guard routes such cases to ARS
     * before they reach this function, so the clip below is a safety net only.
     *
     * Cost vs two-branch version: one bool set once per batch (not per sample),
     * one subtraction per high-Z sample.  Unmeasurable at 10 M/s throughput.
     */
    int    reflect = (chi / nu > 0.5);
    double rate    = reflect ? (nu - chi) : chi;
    for (int i = 0; i < n; i++) {
        double tau = cb_gamma(&s->rng, shape, rate);
        double phi = exp(-tau);            /* positive representable double    */
        double th  = reflect ? (1.0 - phi) : phi;
        if (th <= 0.0) th = 5e-324;        /* low-Z underflow safety           */
        if (th >= 1.0) th = nextafter(1.0, 0.0);/* high-Z safety */
        out[i] = th;
    }
}

static void draw_cf_batch(cb_stream_t *s, int n, double *out)
{
    double eta_star = s->eta_star;
    double sigma    = s->sigma;
    double cf_coef  = s->cf_coef;
    for (int i = 0; i < n; i++) {
        double u  = cb_rng_normal(&s->rng, &s->normal_spare, &s->has_spare);
        double eta = eta_star + sigma*u - cf_coef*sigma*(u*u - 1.0);
        double th  = 1.0 / (1.0 + exp(-eta));
        if (th <= 0.0) th = 5e-324;
        if (th >= 1.0) th = nextafter(1.0, 0.0);
        out[i] = th;
    }
}

static void draw_ars_batch(cb_stream_t *s, int n, double *out)
{
    /* Lazy hull rebuild: if the table is invalid, build it now before drawing.
     * This is the ONLY place build_hull() is called -- never in update().
     * s->chi_eff and s->eta_star already reflect the low-Z parameterisation
     * set by determine_regime() -- see reflection logic there. */
    if (!s->hull_valid) {
        build_hull(s);
    }
    /* Count draws against this hull for adaptive buffer sizing.
     * buf_target for the NEXT refill grows toward buf_capacity as long
     * as the hull remains valid, resetting to CB_DEFAULT_BUF on rebuild. */
    s->draws_since_rebuild += (uint64_t)n;
    {
        int next = (int)(s->draws_since_rebuild) + CB_DEFAULT_BUF;
        s->buf_target = (next < s->buf_capacity) ? next : s->buf_capacity;
    }

    const cb_hull_t *h = &s->hull;
    double chi = s->chi_eff, nu = s->nu;  /* chi_eff: low-Z always */
    int m = h->m, drawn = 0;

    /* Precompute segment probabilities once per batch */
    double probs[CB_ARS_MAX_HULL];
    {
        double mx = h->logwts[0];
        for (int i = 1; i < m; i++) if (h->logwts[i] > mx) mx = h->logwts[i];
        double sum = 0.0;
        for (int i = 0; i < m; i++) { probs[i] = exp(h->logwts[i] - mx); sum += probs[i]; }
        for (int i = 0; i < m; i++) probs[i] /= sum;
    }

    while (drawn < n) {
        int need  = n - drawn;
        int batch = (int)(CB_ARS_OVERBOOK * need);
        if (batch < CB_ARS_MIN_BATCH) batch = CB_ARS_MIN_BATCH;

        for (int b = 0; b < batch && drawn < n; b++) {
            double u = cb_rng_uniform(&s->rng);
            double cdf = 0.0;
            int seg = m - 1;
            for (int i = 0; i < m-1; i++) {
                cdf += probs[i];
                if (u <= cdf) { seg = i; break; }
            }

            double sl  = h->slopes[seg];
            double fv  = h->fvals[seg];
            double pp  = h->pts[seg];
            double a   = (seg == 0)   ? -INFINITY : h->zpts[seg-1];
            double bv  = (seg == m-1) ?  INFINITY  : h->zpts[seg];

            /* Generate proposal ep from the piecewise-exponential envelope.
             * Tail segments (seg=0, seg=m-1) use the generic inversion.
             * Middle segments use the precomputed exm1[seg] = expm1(s*(b-a))
             * to avoid recomputing it on every proposal. */
            double ep;
            if (seg == 0) {
                /* Left tail: s > 0, a = -inf  =>  ep = bv + log(u2)/sl */
                double u2 = cb_rng_uniform(&s->rng);
                ep = bv + log(u2) / sl;
            } else if (seg == m-1) {
                /* Right tail: s < 0, b = +inf  =>  ep = a + log1p(-u2)/sl */
                double u2 = cb_rng_uniform(&s->rng);
                ep = a + log1p(-u2) / sl;
            } else {
                /* Middle segment: use precomputed exm1[seg] = expm1(s*(b-a)).
                 * Guard for |sl| < 1e-8: the center hull point sits exactly at
                 * the posterior mode where slope = chi - nu*B'(eta*) = 0 by
                 * definition.  Fall back to uniform over [a,bv] -- same as
                 * the original sample_trunc_exp() for the zero-slope case. */
                double u2 = cb_rng_uniform(&s->rng);
                if (fabs(sl) < 1e-8)
                    ep = a + u2 * (bv - a);
                else if (sl > 0.0)
                    ep = a + log1p( u2 *  h->exm1[seg]) / sl;
                else
                    ep = a + log1p(-u2 * (-h->exm1[seg])) / sl;
            }
            double log_env = fv + sl * (ep - pp);
            double log_u   = log(cb_rng_uniform(&s->rng));

            /* Chord squeeze */
            int k = 0;
            while (k < m-1 && ep > h->pts[k+1]) k++;
            double log_sq = -INFINITY;
            if (ep > h->pts[0] && ep < h->pts[m-1]) {
                double dp = h->pts[k+1] - h->pts[k];
                double df = h->fvals[k+1] - h->fvals[k];
                log_sq = h->fvals[k] + (df/dp)*(ep - h->pts[k]) - log_env;
            }

            int acc;
            if (log_u <= log_sq) {
                acc = 1;
            } else {
                double log_acc = chi*ep - nu*cb_bft(ep) - log_env;
                if (log_acc > 0.0) log_acc = 0.0;
                acc = (log_u <= log_acc);
            }

            if (acc) {
                double th = 1.0 / (1.0 + exp(-ep));
                if (th <= 0.0) th = 5e-324;
                if (th >= 1.0) th = nextafter(1.0, 0.0);
                out[drawn++] = th;
            }
        }
    }
}

static void draw_core_batch(cb_stream_t *s, int n, double *out)
{
    if (!s->core_valid) return;
    s->draws_since_rebuild += (uint64_t)n;
    {
        int next = (int)(s->draws_since_rebuild) + CB_DEFAULT_BUF;
        s->buf_target = (next < s->buf_capacity) ? next : s->buf_capacity;
    }

    if (s->core_zig_valid) {
        const cb_core_zig_t *z = &s->core_zig;
        int drawn = 0;
        while (drawn < n) {
            double side_u = cb_rng_uniform(&s->rng);
            int right = side_u < z->p_right;
            const double *x = right ? z->x_R : z->x_L;
            const double *y = right ? z->y_R : z->y_L;
            const double *prob = right ? z->R_prob : z->L_prob;
            const int *alias = right ? z->R_alias : z->L_alias;

            double u0 = cb_rng_uniform(&s->rng) * (double)CB_CORE_ZIG_K;
            int k0 = (int)u0;
            if (k0 >= CB_CORE_ZIG_K) k0 = CB_CORE_ZIG_K - 1;
            int k = (u0 - (double)k0 <= prob[k0]) ? k0 : alias[k0];

            double mag = x[k] + cb_rng_uniform(&s->rng) * (x[k+1] - x[k]);
            double yr = cb_rng_uniform(&s->rng) * y[k];
            double u = right ? mag : -mag;

#ifdef CB_CORE_BENCH
            bench_core_zig_proposals++;
#endif
            if (yr <= y[k+1]) {
#ifdef CB_CORE_BENCH
                bench_core_zig_squeeze_accepts++;
#endif
                out[drawn++] = u;
            } else {
#ifdef CB_CORE_BENCH
                bench_core_zig_shape_tests++;
#endif
                double log_acc = core_log_shape(s, u) - z->f0;
                if (log_acc > 0.0) log_acc = 0.0;
                if (log(yr) <= log_acc) {
#ifdef CB_CORE_BENCH
                    bench_core_zig_shape_accepts++;
#endif
                    out[drawn++] = u;
                }
            }
        }
        return;
    }

    int drawn = 0;
    const cb_hull_t *h = &s->core_hull;
    int m = h->m;
    while (drawn < n) {
        double u, log_env;
        sample_core_proposal(s, &u, &log_env);
        double log_u = log(cb_rng_uniform(&s->rng));

        /* Chord squeeze for the standardized core density.  The core shape is
         * concave on its certified support, so chords between tangent points
         * are lower bounds.  Most accepted proposals avoid the K-anchor
         * core_log_shape() evaluation entirely. */
        int acc = 0;
        if (u > h->pts[0] && u < h->pts[m-1]) {
            int k = 0;
            while (k < m - 1 && u > h->pts[k+1]) k++;
            double dp = h->pts[k+1] - h->pts[k];
            double df = h->fvals[k+1] - h->fvals[k];
            double log_sq = h->fvals[k] + (df / dp) * (u - h->pts[k]) - log_env;
            acc = (log_u <= log_sq);
        }
        if (!acc) {
            double log_acc = core_log_shape(s, u) - log_env;
            if (log_acc > 0.0) log_acc = 0.0;
            acc = (log_u <= log_acc);
        }
        if (acc)
            out[drawn++] = u;
    }
}

static void draw_remainder_batch(cb_stream_t *s, int n, double *out)
{
    if (n <= 0) return;
    if (!s->core_valid) {
        draw_ars_batch(s, n, out);
        return;
    }

    int cache_ok = s->rem_valid
        && s->rem_reflect == s->reflect
        && fabs(s->rem_chi_eff - s->chi_eff) <= 1e-14 * (1.0 + fabs(s->chi_eff))
        && fabs(s->rem_nu      - s->nu)      <= 1e-14 * (1.0 + fabs(s->nu))
        && fabs(s->rem_eta_star - s->eta_star) <= 1e-14 * (1.0 + fabs(s->eta_star))
        && fabs(s->rem_sigma    - s->sigma)    <= 1e-14 * (1.0 + fabs(s->sigma));

    double chi = s->chi_eff, nu = s->nu;
    double log_sub_const;
    if (!cache_ok) {
        s->rem_cold_builds++;
        build_hull_exact(s);
        copy_hull(&s->rem_hull, &s->hull);

        double peak = target_log_f(chi, nu, s->eta_star);
        double logW = current_logW_standardized(s);
        double logZq = peak + log(s->sigma) + logW;
        log_sub_const = log(s->core_alpha_hat) + logZq
                      - log(s->sigma) - s->core_logZS;

        s->rem_valid = 1;
        s->rem_reflect = s->reflect;
        s->rem_chi_eff = s->chi_eff;
        s->rem_nu = s->nu;
        s->rem_eta_star = s->eta_star;
        s->rem_sigma = s->sigma;
        s->rem_log_sub_const = log_sub_const;
    } else {
        s->rem_cache_hits++;
        log_sub_const = s->rem_log_sub_const;
    }

    const cb_hull_t *h = &s->rem_hull;
    double probs[CB_ARS_MAX_HULL];
    double mx = h->logwts[0], sum = 0.0;
    for (int i = 1; i < h->m; i++) if (h->logwts[i] > mx) mx = h->logwts[i];
    for (int i = 0; i < h->m; i++) { probs[i] = exp(h->logwts[i] - mx); sum += probs[i]; }

    int drawn = 0;
    while (drawn < n) {
        double eta, log_env;
        hull_proposal_pre(s, probs, sum, &eta, &log_env);
        double lf = target_log_f(chi, nu, eta);
        double u = (eta - s->eta_star) / s->sigma;
        double ls = core_log_shape(s, u);
        double lr = isfinite(ls) ? logdiffexp(lf, log_sub_const + ls) : lf;
        double log_acc = lr - log_env;
        if (log_acc > 0.0) log_acc = 0.0;
        if (log(cb_rng_uniform(&s->rng)) <= log_acc) {
            double th = 1.0 / (1.0 + exp(-eta));
            if (th <= 0.0) th = 5e-324;
            if (th >= 1.0) th = nextafter(1.0, 0.0);
            out[drawn++] = th;
        }
    }
    s->remainder_draws += (uint64_t)n;
    s->hull_valid = 0;
}

/* =========================================================================
 * Buffer refill
 * ========================================================================= */
static void do_refill(cb_stream_t *s, int n_hint)
{
    /* Use the adaptive target for every regime.  It grows when buffers are
     * drained and shrinks when updates invalidate mostly-unused buffers. */
    int n = s->buf_target;
    if (n > n_hint)          n = n_hint;
    if (n > s->buf_capacity) n = s->buf_capacity;
    if (n < 1)               n = 1;

    s->buf_pos    = 0;
    s->buf_filled = 0;
    s->buf_kind   = CB_BUF_EMPTY;
    switch (s->regime) {
    case CB_REGIME_PRIOR:
        for (int i = 0; i < n; i++) s->buf[i] = cb_rng_uniform(&s->rng);
        s->buf_kind = CB_BUF_THETA;
        break;
    case CB_REGIME_GAMMA:
        draw_gamma_batch(s, n, s->buf);
        s->buf_kind = CB_BUF_THETA;
        break;
    case CB_REGIME_CF:
        draw_cf_batch(s, n, s->buf);
        s->buf_kind = CB_BUF_THETA;
        break;
    case CB_REGIME_ARS:
        if (!s->core_valid) build_core(s);
        if (s->core_valid) {
            draw_core_batch(s, n, s->buf);
            s->buf_kind = CB_BUF_CORE_U;
        } else {
            build_hull_exact(s);
            draw_ars_batch(s, n, s->buf);
            s->hull_valid = 0;
            s->buf_kind = CB_BUF_THETA;
        }
        break;
    case CB_REGIME_POINT:
        for (int i = 0; i < n; i++) s->buf[i] = s->reflect ? 1.0 : 0.0;
        s->buf_kind = CB_BUF_THETA;
        break;
    }
    s->buf_filled = n;
    s->total_refills++;

    /* Sign-bit reflection for high-Z arms.
     * The batch draw functions always produce phi = sigmoid(eta) in (0,1)
     * using the reflected chi_eff = nu-chi parameterisation.
     * For high-Z arms (reflect=1), negate the buffer: buf[i] = -phi.
     * At drain time in cb_stream_draw, the transformation is:
     *   theta = v < 0 ? 1.0 + v : v
     * which maps -phi -> 1-phi (= the high-Z draw) and phi -> phi unchanged.
     * This single vectorisable loop replaces all per-sample conditionals.
     * Only applies to exact ARS draws; Gamma and CF handle reflection internally.
     */
    if (s->reflect && (s->regime == CB_REGIME_ARS) && !s->core_valid) {
        for (int i = 0; i < n; i++) s->buf[i] = -s->buf[i];
    }
}

static void adapt_after_drain(cb_stream_t *s)
{
    if (s->buf_filled <= 0 || s->buf_pos < s->buf_filled) return;
    int next = s->buf_filled + CB_DEFAULT_BUF;
    if (next < s->buf_filled) next = s->buf_capacity;  /* overflow guard */
    if (next > s->buf_capacity) next = s->buf_capacity;
    if (next > s->buf_target) s->buf_target = next;
}

static void invalidate_buffer_after_update(cb_stream_t *s)
{
    if (s->buf_filled > 0) {
        int used = s->buf_pos;
        if (used < 1) used = 1;
        if (used < s->buf_target) s->buf_target = used;
    }
    s->buf_pos    = 0;
    s->buf_filled = 0;
    s->buf_kind   = CB_BUF_EMPTY;
}

static void draw_current_batch(cb_stream_t *s, int n, double *out)
{
    switch (s->regime) {
    case CB_REGIME_PRIOR:
        for (int i = 0; i < n; i++) out[i] = cb_rng_uniform(&s->rng);
        break;
    case CB_REGIME_GAMMA:
        draw_gamma_batch(s, n, out);
        break;
    case CB_REGIME_CF:
        draw_cf_batch(s, n, out);
        break;
    case CB_REGIME_ARS:
        build_hull_exact(s);
        draw_ars_batch(s, n, out);
        if (s->reflect) {
            for (int i = 0; i < n; i++) out[i] = 1.0 - out[i];
        }
        s->hull_valid = 0;
        break;
    case CB_REGIME_POINT:
        for (int i = 0; i < n; i++) out[i] = s->reflect ? 1.0 : 0.0;
        break;
    }
}

/* =========================================================================
 * Public API
 * ========================================================================= */
cb_stream_t *cb_stream_create(double chi, double nu,
                              uint64_t base_seed, uint64_t stream_idx,
                              int buf_cap)
{
    if (!cb_stats_valid(chi, nu)) return NULL;
    cb_stream_t *s = (cb_stream_t *)calloc(1, sizeof(cb_stream_t));
    if (!s) return NULL;
    /* Default capacity is the adaptive maximum so buf_target can grow freely.
     * Callers who pass an explicit buf_size get exactly that capacity. */
    if (buf_cap <= 0) buf_cap = CB_MAX_ADAPTIVE_BUF;
    s->buf = (double *)malloc((size_t)buf_cap * sizeof(double));
    if (!s->buf) { free(s); return NULL; }
    s->buf_capacity       = buf_cap;
    s->buf_kind           = CB_BUF_EMPTY;
    s->has_spare          = 0;
    s->mode_valid         = 0;
    s->hull_valid         = 0;   /* force build on first draw */
    s->core_valid         = 0;
    s->core_reflect       = 0;
    s->core_chi0          = 0.0;
    s->core_nu0           = 0.0;
    s->core_R             = 0.0;
    s->core_nu1           = 0.0;
    s->core_logZS         = NAN;
    s->core_alpha_hat     = 0.0;
    s->core_zig_valid     = 0;
    s->rem_valid          = 0;
    s->core_draws         = 0;
    s->remainder_draws    = 0;
    s->core_rebuilds      = 0;
    s->rem_cold_builds    = 0;
    s->rem_cache_hits     = 0;
    s->hull_rebuilds      = 0;
    s->draws_since_rebuild= 0;
    s->buf_target         = CB_DEFAULT_BUF;
    s->chi = chi;
    s->nu  = nu;
    cb_rng_seed(&s->rng, cb_stream_seed(base_seed, stream_idx));
    determine_regime(s);

    /* Pre-fill the buffer at construction time.
     * Pays the hull build cost once upfront so the first draw() is free.
     * For suboptimal arms that are instantiated but rarely selected, this
     * means the arm is always ready to serve immediately when chosen. */
    do_refill(s, s->buf_target);

    return s;
}

void cb_stream_destroy(cb_stream_t *s)
{
    if (!s) return;
    free(s->buf);
    free(s);
}

int cb_stream_update(cb_stream_t *s, double x)
{
    if (!s || !cb_observation_valid(x)) return 0;
    double chi = s->chi + x;
    double nu = s->nu + 1.0;
    if (!cb_stats_valid(chi, nu)) return 0;
    s->chi = chi;
    s->nu  = nu;
    s->rem_valid  = 0;
    determine_regime(s);
    if (!(s->regime == CB_REGIME_ARS
          && s->buf_kind == CB_BUF_CORE_U
          && core_covers_current(s))) {
        invalidate_buffer_after_update(s);
    }
    return 1;
}

int cb_stream_update_batch(cb_stream_t *s, const double *xs, int n)
{
    if (!s || n < 0 || (n > 0 && !xs)) return 0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (!cb_observation_valid(xs[i])) return 0;
        sum += xs[i];
    }
    double chi = s->chi + sum;
    double nu = s->nu + (double)n;
    if (!cb_stats_valid(chi, nu)) return 0;
    s->chi = chi;
    s->nu  = nu;
    s->rem_valid  = 0;
    determine_regime(s);
    if (!(s->regime == CB_REGIME_ARS
          && s->buf_kind == CB_BUF_CORE_U
          && core_covers_current(s))) {
        invalidate_buffer_after_update(s);
    }
    return 1;
}

int cb_stream_set_stats(cb_stream_t *s, double chi, double nu)
{
    if (!s || !cb_stats_valid(chi, nu)) return 0;
    s->chi = chi;
    s->nu  = nu;
    s->rem_valid = 0;
    determine_regime(s);
    if (!(s->regime == CB_REGIME_ARS
          && s->buf_kind == CB_BUF_CORE_U
          && core_covers_current(s))) {
        invalidate_buffer_after_update(s);
    }
    return 1;
}

int cb_stream_draw(cb_stream_t *s, int n, double *out)
{
    if (!s || n < 0 || (n > 0 && !out)
            || !cb_stats_valid(s->chi, s->nu)) return -1;
    if (n > s->buf_capacity) {
        /* Large requests outgrow the stream buffer.  Draw the whole request
         * from the current exact batch path, then discard/refill the buffer
         * so future small requests again amortize the stream core.  This keeps
         * correctness simple: no partially reused buffer is mixed with a
         * parameter-specific batch draw. */
        s->buf_pos = 0;
        s->buf_filled = 0;
        s->buf_kind = CB_BUF_EMPTY;
        draw_current_batch(s, n, out);
        cb_clip_interior_draws(s, n, out);
        s->total_draws += (uint64_t)n;
        return n;
    }

    if (n == 1) {
        if (s->buf_pos >= s->buf_filled)
            do_refill(s, s->buf_capacity);
        if (s->buf_filled == 0) return 0;

        if (s->regime == CB_REGIME_ARS && s->core_valid && s->buf_kind == CB_BUF_CORE_U) {
            double p_rem = 1.0 - s->core_alpha_hat;
            if (cb_rng_uniform(&s->rng) < p_rem) {
                double rem;
                draw_remainder_batch(s, 1, &rem);
                out[0] = s->reflect ? 1.0 - rem : rem;
            } else {
                double u = s->buf[s->buf_pos++];
                double eta = s->eta_star + s->sigma * u;
                double th = 1.0 / (1.0 + exp(-eta));
                if (th <= 0.0) th = 5e-324;
                if (th >= 1.0) th = nextafter(1.0, 0.0);
                out[0] = s->reflect ? 1.0 - th : th;
                s->core_draws++;
            }
            cb_clip_interior_draws(s, 1, out);
            adapt_after_drain(s);
            s->total_draws++;
            return 1;
        }

        {
            double v = s->buf[s->buf_pos++];
            out[0] = v < 0.0 ? 1.0 + v : v;
            cb_clip_interior_draws(s, 1, out);
            adapt_after_drain(s);
            s->total_draws++;
            return 1;
        }
    }

    int drawn = 0;
    while (drawn < n) {
        if (s->buf_pos >= s->buf_filled)
            do_refill(s, s->buf_capacity);
        if (s->buf_filled == 0) break;
        int avail = s->buf_filled - s->buf_pos;
        int take  = (n - drawn < avail) ? (n - drawn) : avail;

        if (s->regime == CB_REGIME_ARS && s->core_valid && s->buf_kind == CB_BUF_CORE_U) {
            double alpha = s->core_alpha_hat;
            double p_rem = 1.0 - alpha;
            int k_rem = (take == 1)
                      ? (cb_rng_uniform(&s->rng) < p_rem)
                      : binom_wait2_rem(&s->rng, take, p_rem);
            int k_core = take - k_rem;
            int block0 = drawn;

            if (k_rem > 0) {
                double *rem = (double *)malloc((size_t)k_rem * sizeof(double));
                if (!rem) return drawn;
                draw_remainder_batch(s, k_rem, rem);
                if (s->reflect) {
                    for (int i = 0; i < k_rem; i++)
                        rem[i] = 1.0 - rem[i];
                }

                const double *src = s->buf + s->buf_pos;
                int rem_left = k_rem, rem_i = 0, core_i = 0;
                for (int i = 0; i < take; i++) {
                    int slots_left = take - i;
                    int use_rem = cb_rng_uniform(&s->rng) * (double)slots_left
                                  < (double)rem_left;
                    if (use_rem) {
                        out[block0 + i] = rem[rem_i++];
                        rem_left--;
                    } else {
                        double u = src[core_i++];
                        double eta = s->eta_star + s->sigma * u;
                        double th = 1.0 / (1.0 + exp(-eta));
                        if (th <= 0.0) th = 5e-324;
                        if (th >= 1.0) th = nextafter(1.0, 0.0);
                        out[block0 + i] = s->reflect ? 1.0 - th : th;
                    }
                }
                free(rem);
            } else {
                const double *src = s->buf + s->buf_pos;
                double *dst = out + drawn;
                for (int i = 0; i < k_core; i++) {
                    double eta = s->eta_star + s->sigma * src[i];
                    double th = 1.0 / (1.0 + exp(-eta));
                    if (th <= 0.0) th = 5e-324;
                    if (th >= 1.0) th = nextafter(1.0, 0.0);
                    dst[i] = s->reflect ? 1.0 - th : th;
                }
            }
            s->buf_pos += k_core;
            adapt_after_drain(s);
            drawn += take;
            s->core_draws += (uint64_t)k_core;
            continue;
        }

        /* Sign-bit decode: v < 0 means reflected high-Z draw, theta = 1 + v.
         * v >= 0 means low-Z draw, theta = v.  Branchless, vectorisable. */
        const double *src = s->buf + s->buf_pos;
        double       *dst = out + drawn;
        for (int i = 0; i < take; i++) {
            double v = src[i];
            dst[i]   = v < 0.0 ? 1.0 + v : v;
        }
        s->buf_pos += take;
        adapt_after_drain(s);
        drawn      += take;
    }
    cb_clip_interior_draws(s, drawn, out);
    s->total_draws += (uint64_t)drawn;
    return drawn;
}

void cb_stream_peek(const cb_stream_t *s,
                    double *chi_out, double *nu_out,
                    int *regime_out, double *sigma_out)
{
    if (chi_out)    *chi_out    = s->chi;
    if (nu_out)     *nu_out     = s->nu;
    if (regime_out) *regime_out = s->regime;
    if (sigma_out)  *sigma_out  = s->sigma;
}

/* Return hull rebuild count since stream creation (ARS diagnostic). */
uint64_t cb_stream_hull_rebuilds(const cb_stream_t *s)
{
    return s->hull_rebuilds;
}

void cb_stream_core_stats(const cb_stream_t *s,
                          double *alpha_hat,
                          uint64_t *core_draws,
                          uint64_t *remainder_draws,
                          uint64_t *rebuilds,
                          int *core_active,
                          uint64_t *rem_cold_builds,
                          uint64_t *rem_cache_hits)
{
    if (alpha_hat)       *alpha_hat       = s->core_valid ? s->core_alpha_hat : 0.0;
    if (core_draws)      *core_draws      = s->core_draws;
    if (remainder_draws) *remainder_draws = s->remainder_draws;
    if (rebuilds)        *rebuilds        = s->core_rebuilds;
    if (core_active)     *core_active     = s->core_valid ? 1 : 0;
    if (rem_cold_builds) *rem_cold_builds = s->rem_cold_builds;
    if (rem_cache_hits)  *rem_cache_hits  = s->rem_cache_hits;
}

int cb_sample_c(double chi, double nu, uint64_t seed, int n, double *out)
{
    cb_stream_t *s = cb_stream_create(chi, nu, seed, 0, n > 0 ? n : CB_DEFAULT_BUF);
    if (!s) return -1;
    int drawn = cb_stream_draw(s, n, out);
    cb_stream_destroy(s);
    return drawn;
}

/* =========================================================================
 * Standalone test
 * ========================================================================= */
#if defined(CB_CORE_TEST) || defined(CB_CORE_BENCH)
#include <stdio.h>
#include <time.h>
static const char *rname(int r){
    switch(r){case 0:return"PRIOR";case 1:return"GAMMA";case 2:return"CF";case 3:return"ARS";case 4:return"POINT";}
    return"?";
}
#endif

#ifdef CB_CORE_BENCH
static double wall_seconds(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void reset_core_bench_counters(void)
{
    bench_core_zig_proposals = 0;
    bench_core_zig_squeeze_accepts = 0;
    bench_core_zig_shape_tests = 0;
    bench_core_zig_shape_accepts = 0;
}

static void print_core_bench_counters(const char *label)
{
    double p = (double)bench_core_zig_proposals;
    double sq = (double)bench_core_zig_squeeze_accepts;
    double st = (double)bench_core_zig_shape_tests;
    double sa = (double)bench_core_zig_shape_accepts;
    double acc = (sq + sa) / (p > 0.0 ? p : 1.0);
    printf("%s core-zig: proposals=%llu acc/proposal=%.4f squeeze=%.4f shape_test=%.4f shape_acc=%.4f\n",
           label,
           (unsigned long long)bench_core_zig_proposals,
           acc,
           sq / (p > 0.0 ? p : 1.0),
           st / (p > 0.0 ? p : 1.0),
           sa / (st > 0.0 ? st : 1.0));
}

static void print_core_stats(cb_stream_t *s, double draws, double seconds)
{
    double alpha = 0.0;
    uint64_t core_draws = 0, rem_draws = 0, rebuilds = 0, rem_cold = 0, rem_hits = 0;
    int active = 0;
    cb_stream_core_stats(s, &alpha, &core_draws, &rem_draws, &rebuilds, &active,
                         &rem_cold, &rem_hits);
    double total_core = (double)(core_draws + rem_draws);
    double core_frac = total_core > 0.0 ? (double)core_draws / total_core : 0.0;
    printf("%12.3g %10.3f %10.3f %9llu %8d",
           seconds > 0.0 ? draws / seconds : 0.0,
           alpha,
           core_frac,
           (unsigned long long)rebuilds,
           active);
}

int main(void)
{
    enum { N_BULK = 1000000, T_HOT = 200000, T_REM = 50000 };
    const int bufs[] = {64, 128, 256, 512, 1024, 4096};
    const int nbuf = (int)(sizeof(bufs) / sizeof(bufs[0]));
    struct {
        const char *name;
        double chi, nu, x;
    } cases[] = {
        {"extreme-hot", 15.0,  300.0,  0.05},
        {"interior-ars",120.0, 300.0,  0.40},
        {"near-cf",     480.0, 1000.0, 0.48}
    };
    const int ncases = (int)(sizeof(cases) / sizeof(cases[0]));
    double *out = (double *)malloc((size_t)N_BULK * sizeof(double));
    if (!out) return 2;

    printf("C-only continuous Bernoulli stream benchmark\n");
    printf("N_bulk=%d, T_hot=%d\n\n", N_BULK, T_HOT);

    printf("Fixed posterior stream draw\n");
    printf("%-12s %6s %8s %12s %10s %10s %9s %8s\n",
           "case", "buf", "regime", "draws/sec", "alpha", "core_frac", "rebuilds", "active");
    reset_core_bench_counters();
    for (int c = 0; c < ncases; c++) {
        for (int b = 0; b < nbuf; b++) {
            cb_stream_t *s = cb_stream_create(cases[c].chi, cases[c].nu,
                                              1000u + (uint64_t)(17*c + b),
                                              (uint64_t)b, bufs[b]);
            double t0 = wall_seconds();
            int left = N_BULK, pos = 0;
            while (left > 0) {
                int chunk = bufs[b] < 1024 ? bufs[b] : 1024;
                if (chunk > left) chunk = left;
                cb_stream_draw(s, chunk, out + pos);
                pos += chunk;
                left -= chunk;
            }
            double dt = wall_seconds() - t0;
            printf("%-12s %6d %8s ", cases[c].name, bufs[b], rname(s->regime));
            print_core_stats(s, (double)N_BULK, dt);
            printf("\n");
            cb_stream_destroy(s);
        }
    }
    print_core_bench_counters("fixed");

    printf("\nDirect core u-fill and transport probes\n");
    printf("%-18s %12s\n", "probe", "items/sec");
    {
        cb_stream_t *s = cb_stream_create(15.0, 300.0, 7777u, 0, 4096);
        if (!s->core_valid) build_core(s);
        reset_core_bench_counters();
        double t0 = wall_seconds();
        draw_core_batch(s, N_BULK, out);
        double dt = wall_seconds() - t0;
        printf("%-18s %12.3g\n", "core_u_fill", dt > 0.0 ? (double)N_BULK / dt : 0.0);
        print_core_bench_counters("direct core");
        cb_stream_destroy(s);
    }
    {
        cb_stream_t *s = cb_stream_create(15.0, 300.0, 8888u, 0, 4096);
        if (!s->core_valid) build_core(s);
        draw_core_batch(s, N_BULK, out);
        double eta_star = s->eta_star, sigma = s->sigma;
        double t0 = wall_seconds();
        for (int i = 0; i < N_BULK; i++) {
            double eta = eta_star + sigma * out[i];
            double th = 1.0 / (1.0 + exp(-eta));
            if (th <= 0.0) th = 5e-324;
            if (th >= 1.0) th = nextafter(1.0, 0.0);
            out[i] = th;
        }
        double dt = wall_seconds() - t0;
        printf("%-18s %12.3g\n", "u_to_theta", dt > 0.0 ? (double)N_BULK / dt : 0.0);
        cb_stream_destroy(s);
    }

    printf("\nUpdate/draw split probes\n");
    printf("%-18s %10s %12s\n", "probe", "us/iter", "iters/sec");
    {
        cb_stream_t *s = cb_stream_create(15.0, 300.0, 9001u, 0, 4096);
        double t0 = wall_seconds();
        for (int k = 0; k < T_HOT; k++)
            cb_stream_update(s, 0.05);
        double dt = wall_seconds() - t0;
        printf("%-18s %10.3f %12.3g\n", "update_only",
               dt > 0.0 ? 1e6 * dt / (double)T_HOT : 0.0,
               dt > 0.0 ? (double)T_HOT / dt : 0.0);
        cb_stream_destroy(s);
    }
    {
        cb_stream_t *s = cb_stream_create(15.0, 300.0, 9002u, 0, T_HOT + 1024);
        double t0 = wall_seconds();
        for (int k = 0; k < T_HOT; k++)
            cb_stream_draw(s, 1, out);
        double dt = wall_seconds() - t0;
        printf("%-18s %10.3f %12.3g\n", "draw1_stable",
               dt > 0.0 ? 1e6 * dt / (double)T_HOT : 0.0,
               dt > 0.0 ? (double)T_HOT / dt : 0.0);
        cb_stream_destroy(s);
    }
    {
        cb_stream_t *s = cb_stream_create(15.0, 300.0, 9003u, 0, T_HOT + 1024);
        if (s->core_valid) s->core_alpha_hat = 1.0;
        double t0 = wall_seconds();
        for (int k = 0; k < T_HOT; k++)
            cb_stream_draw(s, 1, out);
        double dt = wall_seconds() - t0;
        printf("%-18s %10.3f %12.3g\n", "draw1_core_only",
               dt > 0.0 ? 1e6 * dt / (double)T_HOT : 0.0,
               dt > 0.0 ? (double)T_HOT / dt : 0.0);
        cb_stream_destroy(s);
    }

    printf("\nIn-C update/draw(1) loop\n");
    printf("%-12s %6s %8s %10s %12s %10s %10s %9s %8s\n",
           "case", "buf", "regime", "us/iter", "draws/sec", "alpha", "core_frac", "rebuilds", "active");
    reset_core_bench_counters();
    for (int c = 0; c < ncases; c++) {
        for (int b = 0; b < nbuf; b++) {
            cb_stream_t *s = cb_stream_create(cases[c].chi, cases[c].nu,
                                              2000u + (uint64_t)(17*c + b),
                                              (uint64_t)b, bufs[b]);
            int reg0 = s->regime;
            double t0 = wall_seconds();
            for (int k = 0; k < T_HOT; k++) {
                cb_stream_update(s, cases[c].x);
                cb_stream_draw(s, 1, out);
            }
            double dt = wall_seconds() - t0;
            printf("%-12s %6d %8s %10.3f ", cases[c].name, bufs[b], rname(reg0),
                   dt > 0.0 ? 1e6 * dt / (double)T_HOT : 0.0);
            print_core_stats(s, (double)T_HOT, dt);
            printf("\n");
            cb_stream_destroy(s);
        }
    }
    print_core_bench_counters("update/draw");

    printf("\nLarge-request bypass\n");
    printf("%-12s %6s %8s %12s\n", "case", "N", "regime", "draws/sec");
    for (int c = 0; c < ncases; c++) {
        cb_stream_t *s = cb_stream_create(cases[c].chi, cases[c].nu,
                                          3000u + (uint64_t)c,
                                          (uint64_t)c, 256);
        double t0 = wall_seconds();
        cb_stream_draw(s, N_BULK, out);
        double dt = wall_seconds() - t0;
        printf("%-12s %6d %8s %12.3g\n", cases[c].name, N_BULK, rname(s->regime),
               dt > 0.0 ? (double)N_BULK / dt : 0.0);
        cb_stream_destroy(s);
    }

    printf("\nDirect remainder sampler stress\n");
    printf("%-12s %10s %12s\n", "mode", "us/rem", "rem/sec");
    {
        cb_stream_t *s = cb_stream_create(15.0, 300.0, 4444u, 0, 256);
        double t0 = wall_seconds();
        for (int i = 0; i < T_REM; i++) {
            s->rem_valid = 0;
            draw_remainder_batch(s, 1, out);
        }
        double dt = wall_seconds() - t0;
        printf("%-12s %10.3f %12.3g\n", "cold", 1e6 * dt / (double)T_REM,
               dt > 0.0 ? (double)T_REM / dt : 0.0);
        cb_stream_destroy(s);
    }
    {
        cb_stream_t *s = cb_stream_create(15.0, 300.0, 5555u, 0, 256);
        double t0 = wall_seconds();
        for (int i = 0; i < T_REM; i++)
            draw_remainder_batch(s, 1, out);
        double dt = wall_seconds() - t0;
        printf("%-12s %10.3f %12.3g\n", "cached", 1e6 * dt / (double)T_REM,
               dt > 0.0 ? (double)T_REM / dt : 0.0);
        cb_stream_destroy(s);
    }

    free(out);
    return 0;
}
#endif

#ifdef CB_CORE_TEST
int main(void){
    printf("=== cb_core test (CB_SIGMA_CF=0.20) ===\n\n");
    double etas[]={-10,-5,-1.5,-0.5,0.001,0.5,1.5,5,10};
    int ok=1;
    printf("B'' positivity: ");
    for(int i=0;i<9;i++){double d=cb_bft_d2(etas[i]);if(d<=0){printf("FAIL %.1f ",etas[i]);ok=0;}}
    printf("%s\n\n",ok?"OK":"FAIL");
    struct{double chi,nu;int er;double em;const char*d;}C[]={
        {1,    10000, CB_REGIME_GAMMA, 0.000,   "Gamma low "},
        {500,  1000,  CB_REGIME_CF,    0.500,   "CF symm   "},
        {4,    40,    CB_REGIME_ARS,   1.06e-4, "ARS low   "},
        {18,   40,    CB_REGIME_ARS,   0.360,   "ARS mod   "},
        {100,  500,   CB_REGIME_ARS,   0.0082,  "ARS(was CF)"},
    };
    int N=100000;double*out=(double*)malloc((size_t)N*sizeof(double));
    printf("%-14s %-5s %-8s %-8s %-12s\n","Case","Reg","Mean","Expect","Samp/s");
    for(int c=0;c<5;c++){
        cb_stream_t*s=cb_stream_create(C[c].chi,C[c].nu,0x123456789ABCULL,(uint64_t)c,1024);
        clock_t t0=clock();cb_stream_draw(s,N,out);double dt=(double)(clock()-t0)/CLOCKS_PER_SEC;
        double mn=0;for(int i=0;i<N;i++)mn+=out[i];mn/=N;
        int rok=s->regime==C[c].er,mok=fabs(mn-C[c].em)<0.05*(fabs(C[c].em)+1e-5);
        printf("%-14s %-5s %-8.5f %-8.4f %-12.0f %s%s\n",
            C[c].d,rname(s->regime),mn,C[c].em,dt>0?N/dt:1e9,
            rok?"":"[REG!]",mok?"":"[MEAN!]");
        cb_stream_destroy(s);}
    free(out);printf("\nDone.\n");return 0;
}
#endif
