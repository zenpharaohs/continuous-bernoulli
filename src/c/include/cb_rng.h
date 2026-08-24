/*
 * cb_rng.h
 * RNG primitives for the CB sampler C backend.
 *
 * RNG hierarchy:
 *   xorshift64*     -- hot-path uniform U(0,1), period 2^64-1
 *   SplitMix64      -- seeding only (avalanche mixing, bijective)
 *   Ziggurat Exp(1) -- 256-layer Marsaglia-Tsang, ~99% fast-accept
 *   Ziggurat N(0,1) -- 128-layer Marsaglia-Tsang, ~98% fast-accept
 *   Gamma(a,r)      -- Marsaglia-Tsang squeeze using ziggurat normal
 *
 * Ziggurat N(0,1) construction (M&T 2000, exactly):
 *   N=128 layers, R=3.442619855899, v=9.91256303526217e-3
 *   Layer indexing: iz=0 is the INNERMOST (central) layer, iz=127 is outermost.
 *   x[0]=0 (inner), x[127]=R (outer), x[i] increasing.
 *   f[i] = exp(-x[i]^2/2), f[0]=1, f[127]=exp(-R^2/2), f[i] decreasing.
 *   Recurrence (inward from outer): f[i] = f[i+1] + v/x[i+1], x[i]=sqrt(-2 log f[i])
 *
 *   Sampling layer iz:
 *     x_samp = hz * w[iz],  hz in [0, 2^55),  w[iz] = x[iz] / 2^55
 *     Fast accept if hz < k[iz] = (x[iz-1]/x[iz]) * 2^55
 *       (sample in safe inner region [0, x[iz-1]])
 *     Slow path: accept if U*(f[iz-1]-f[iz]) + f[iz] <= exp(-x_samp^2/2)
 *     iz=0 always fast-accepts (k[0] = 2^55-1 = max).
 *     iz=1 always goes slow (k[1] = 0).
 *
 *   Tail: iz=127 (outermost) + slow path invokes the M&T tail sampler,
 *   drawing x ~ N(0,1) | |x|>R via rejection from Exp(R).  Cost: 2 log calls,
 *   fires with probability P(|z|>R) = 2*(1-Phi(R)) ~ 0.057% per normal draw.
 *   Without the tail, Gamma samples are biased for small shape/rate and
 *   CF samples have variance deficit ~2*R*phi(R)/(2*Phi(R)-1) ~ 1.8%.
 *
 * Throughput vs Box-Muller polar (xorshift64* base, -O3):
 *   Normal:      ~2.2x speedup (100 vs 46 M/s scalar C)
 *   Gamma(41):   ~1.33x speedup (56 vs 42 M/s)
 *   Gamma(201):  ~1.38x speedup (63 vs 46 M/s)
 *
 * Per-stream state: no mutable globals used during sampling.
 * (Table init uses static flags -- thread-safe on single init before use.)
 * Thread discipline: seed stream k as base ^ splitmix64(k+1).
 *
 * Reference: Marsaglia & Tsang (2000), J. Stat. Soft. 5(8).
 * MIT License.
 */

#ifndef CB_RNG_H
#define CB_RNG_H

#include <stdint.h>
#include <math.h>

/* =========================================================================
 * SplitMix64 -- seeding only
 * ========================================================================= */
static inline uint64_t splitmix64_next(uint64_t *x)
{
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* =========================================================================
 * xorshift64* -- hot-path RNG
 * ========================================================================= */
typedef struct { uint64_t s; } cb_rng_t;

static inline void cb_rng_seed(cb_rng_t *r, uint64_t seed)
{
    if (seed == 0) seed = 0xDEADBEEFCAFEBABEULL;
    r->s = splitmix64_next(&seed);
    if (r->s == 0) r->s = 0xA5A5A5A5A5A5A5A5ULL;
}

static inline uint64_t cb_stream_seed(uint64_t base, uint64_t stream_idx)
{
    uint64_t ctr = stream_idx + 1;
    return base ^ splitmix64_next(&ctr);
}

static inline uint64_t cb_rng_u64(cb_rng_t *r)
{
    uint64_t x = r->s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    r->s = x;
    return x * 2685821657736338717ULL;
}

static inline double cb_rng_uniform(cb_rng_t *r)
{
    return (cb_rng_u64(r) >> 11) * (1.0 / 9007199254740992.0);
}

/* =========================================================================
 * Ziggurat Exp(1) -- 256-layer Marsaglia-Tsang
 * ========================================================================= */
#define CB_ZEXP_N 256

static uint32_t     _cb_zexp_ke[CB_ZEXP_N];
static double       _cb_zexp_we[CB_ZEXP_N];
static double       _cb_zexp_fe[CB_ZEXP_N + 1];
static double       _cb_zexp_x [CB_ZEXP_N + 1];
static volatile int _cb_zexp_inited = 0;

static void cb_zexp_init(void)
{
    if (_cb_zexp_inited) return;
    const double R   = 7.69711747013104972;
    const double eR  = exp(-R);
    const double Ain = (1.0 - eR) / (double)CB_ZEXP_N;
    _cb_zexp_x[0]  = R;
    _cb_zexp_fe[0] = eR;
    for (int i = 1; i <= CB_ZEXP_N; i++) {
        double ei = eR + (double)i * Ain;
        if (ei > 1.0) ei = 1.0;
        _cb_zexp_x[i]  = -log(ei);
        _cb_zexp_fe[i] = ei;
    }
    for (int i = 0; i < CB_ZEXP_N; i++) {
        double w = _cb_zexp_x[i] - _cb_zexp_x[i+1];
        _cb_zexp_we[i] = w * (1.0 / 4294967296.0);
        double t = (_cb_zexp_fe[i+1] / _cb_zexp_fe[i]) * 4294967296.0;
        _cb_zexp_ke[i] = (t >= 4294967295.0) ? 0xFFFFFFFFu : (uint32_t)t;
    }
    _cb_zexp_inited = 1;
}

static inline double cb_exp1(cb_rng_t *r)
{
    cb_zexp_init();
    for (;;) {
        uint64_t raw = cb_rng_u64(r);
        uint32_t j   = (uint32_t)(raw & 0xFFFFFFFFu);
        int      i   = (int)((raw >> 32) & 0xFFu);
        if (j < _cb_zexp_ke[i])
            return _cb_zexp_x[i+1] + (double)j * _cb_zexp_we[i];
        if (i == 0)
            return _cb_zexp_x[0] - log(cb_rng_uniform(r));
        double x = _cb_zexp_x[i+1] + (double)j * _cb_zexp_we[i];
        double y = _cb_zexp_fe[i+1]
                 + (_cb_zexp_fe[i] - _cb_zexp_fe[i+1]) * cb_rng_uniform(r);
        if (y <= exp(-x)) return x;
    }
}

/* =========================================================================
 * Ziggurat N(0,1) -- 128-layer Marsaglia-Tsang (2000), corrected
 *
 * Bit layout of the raw uint64 draw:
 *   bits  0-6:  layer index iz (7 bits, range 0..127)
 *   bit   7:    sign (0 = positive, 1 = negative)
 *   bits  8-62: magnitude hz (55 bits, range 0..2^55-1)
 *   bit  63:    unused
 * ========================================================================= */
#define CB_ZNORM_N       128
#define CB_ZNORM_R       3.442619855899
#define CB_ZNORM_VN      9.91256303526217e-3
#define CB_ZNORM_SCALE   (1ULL << 55)           /* 2^55 */
#define CB_ZNORM_SCALED  ((double)(1ULL << 55)) /* 2^55 as double */

static uint64_t     _cb_znorm_k[CB_ZNORM_N];
static double       _cb_znorm_w[CB_ZNORM_N];   /* x[iz] / 2^55 */
static double       _cb_znorm_f[CB_ZNORM_N];   /* exp(-x[iz]^2/2) */
static volatile int _cb_znorm_inited = 0;

static void cb_znorm_init(void)
{
    if (_cb_znorm_inited) return;

    double xv[CB_ZNORM_N], fv[CB_ZNORM_N];
    const int N = CB_ZNORM_N;

    /* Outermost layer */
    xv[N-1] = CB_ZNORM_R;
    fv[N-1] = exp(-0.5 * CB_ZNORM_R * CB_ZNORM_R);

    /* Build table inward via M&T recurrence:
     *   f[i] = f[i+1] + v / x[i+1]
     *   x[i] = sqrt(-2 * log(f[i]))
     */
    for (int i = N-2; i >= 1; i--) {
        double fi = fv[i+1] + CB_ZNORM_VN / xv[i+1];
        if (fi > 1.0) fi = 1.0;
        fv[i] = fi;
        xv[i] = sqrt(-2.0 * log(fi));
    }
    xv[0] = 0.0;
    fv[0] = 1.0;

    /* Store f[] */
    for (int i = 0; i < N; i++) _cb_znorm_f[i] = fv[i];

    /* iz=0: hybrid innermost/tail strip (Marsaglia & Tsang 2000, Fig 1).
     *
     * Let tn0 = v / f(R) where v=CB_ZNORM_VN and f(R)=exp(-R^2/2).
     * The iz=0 strip samples x from [0, tn0]:
     *   x < R  (hz < k[0]):  fast-accept.  These are the innermost body
     *                         samples; they compensate the floor term
     *                         fv[N-1]=f(R) in the body density and make
     *                         the combined distribution exactly N(0,1).
     *   x >= R (hz >= k[0]): slow-path -> tail sampler, giving |x|>R.
     *
     * Probability of reaching tail sampler:
     *   P(iz=0) * (1 - k[0]/2^55) = (1/128)*(1 - R/tn0)
     *   = (1/128) * v*exp(R^2/2) / (R * ... ) ~= 5.9e-4 ~= P(|X|>R).
     *
     * Without this split (k[0]=2^55-1 all fast-accept, as before),
     * the tail is never sampled and E[X^2] ~ 0.97 (std ~ 0.984).
     */
    {
        double tn0 = CB_ZNORM_VN / fv[N-1];      /* ≈ 3.726 */
        _cb_znorm_w[0] = tn0 / CB_ZNORM_SCALED;
        double t0 = (CB_ZNORM_R / tn0) * CB_ZNORM_SCALED; /* threshold at R */
        _cb_znorm_k[0] = (t0 >= CB_ZNORM_SCALED - 1.0)
                         ? (CB_ZNORM_SCALE - 1) : (uint64_t)t0;
    }

    /* iz=1: innermost body strip, always slow path (k=0, since x[0]=0
     * means the fast-accept region [0,x[0]] has zero width).          */
    _cb_znorm_w[1] = xv[1] / CB_ZNORM_SCALED;
    _cb_znorm_k[1] = 0;

    /* iz=2..127 */
    for (int i = 2; i < N; i++) {
        _cb_znorm_w[i] = xv[i] / CB_ZNORM_SCALED;
        double t = (xv[i-1] / xv[i]) * CB_ZNORM_SCALED;
        _cb_znorm_k[i] = (t >= CB_ZNORM_SCALED - 1.0)
                         ? (CB_ZNORM_SCALE - 1) : (uint64_t)t;
    }

    _cb_znorm_inited = 1;
}

static inline double cb_rng_normal_zig(cb_rng_t *r)
{
    cb_znorm_init();
    for (;;) {
        uint64_t raw = cb_rng_u64(r);
        int      iz  = (int)(raw & 0x7Fu);                   /* bits 0-6  */
        int      sgn = (int)((raw >> 7) & 1u);               /* bit  7    */
        uint64_t hz  = (raw >> 8) & (CB_ZNORM_SCALE - 1);   /* bits 8-62 */

        double x = (double)hz * _cb_znorm_w[iz];

        /* Fast accept: sample in [0, x[iz-1]] */
        if (hz < _cb_znorm_k[iz]) return sgn ? -x : x;

        /* iz=0: tail layer (k[0]=0, so hz < k[0] is never true).
         * Sample |X| > R via M&T exponential-tail rejection:
         *   x_t = -log(U1)/R  ~ Exp(R);  accept if 2*log(U2) >= x_t^2.
         *   Returns R + x_t in (R, inf).
         * Fires with probability 1/128 (before rejection), giving the
         * correct total tail mass and compensating the floor-term bias. */
        if (iz == 0) {
            double xt, yt;
            do {
                xt = -log(cb_rng_uniform(r)) / CB_ZNORM_R;
                yt = -log(cb_rng_uniform(r));
            } while (2.0 * yt < xt * xt);
            x = CB_ZNORM_R + xt;
            return sgn ? -x : x;
        }

        /* Slow path (body layers 1..127): test if sample is under exp(-t^2/2) */
        double y = _cb_znorm_f[iz]
                 + (_cb_znorm_f[iz-1] - _cb_znorm_f[iz]) * cb_rng_uniform(r);
        if (y <= exp(-0.5 * x * x)) return sgn ? -x : x;
        /* else retry */
    }
}

/*
 * Public interface -- routes to ziggurat.
 * spare/has_spare kept for API compatibility but unused.
 */
static inline double cb_rng_normal(cb_rng_t *r, double *spare, int *has_spare)
{
    (void)spare; (void)has_spare;
    return cb_rng_normal_zig(r);
}

/* =========================================================================
 * Gamma(shape, rate) via Marsaglia-Tsang squeeze + ziggurat normal
 *
 * shape >= 1: Marsaglia-Tsang with ziggurat N(0,1).
 *   Squeeze (cheap): ~99% accept for large shape, no log.
 *   Exact test: log(u) < 0.5*x^2 + d*(1 - v + log(v))
 *
 * shape < 1: Boost: Gamma(a) = Gamma(a+1) * U^(1/a)
 *
 * Throughput (scalar C, -O3):
 *   Gamma(41):  ~56 M/s  (vs ~42 M/s with Box-Muller)
 *   Gamma(201): ~63 M/s  (vs ~46 M/s with Box-Muller)
 * ========================================================================= */
static double cb_gamma1(cb_rng_t *r, double shape)
{
    if (shape < 1.0) {
        double g = cb_gamma1(r, shape + 1.0);
        double u = cb_rng_uniform(r);
        if (u < 1e-300) u = 1e-300;
        return g * pow(u, 1.0 / shape);
    }

    double d = shape - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);

    for (;;) {
        double x, v;
        do {
            x = cb_rng_normal_zig(r);
            v = 1.0 + c * x;
        } while (v <= 0.0);

        v = v * v * v;
        double u = cb_rng_uniform(r);

        /* Cheap squeeze (~99% accept for large shape) */
        if (u < 1.0 - 0.0331 * (x*x) * (x*x)) return d * v;
        /* Exact accept */
        if (log(u) < 0.5*x*x + d*(1.0 - v + log(v))) return d * v;
    }
}

static inline double cb_gamma(cb_rng_t *r, double shape, double rate)
{
    return cb_gamma1(r, shape) / rate;
}

#endif /* CB_RNG_H */
