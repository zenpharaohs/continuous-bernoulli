/*
 * cb_stream_mex.c
 * MEX wrapper for the CB conjugate posterior stream sampler.
 *
 * Commands (string dispatch):
 *
 *   ptr = cb_stream_mex('create', chi, nu, base_seed, stream_idx, buf_cap)
 *     Create a stream.  Returns uint64 pointer handle.
 *     base_seed, stream_idx: uint64 scalars.  buf_cap: int (0 = default 256).
 *
 *   theta = cb_stream_mex('draw', ptr, n)
 *     Draw n samples from the stream.  Returns n-by-1 double vector.
 *
 *   cb_stream_mex('update', ptr, x)
 *     Ingest one observation x in [0,1].  Updates chi, nu, invalidates buffer,
 *     and re-determines the regime.  Boundary posteriors are point masses.
 *
 *   cb_stream_mex('update_batch', ptr, xs)
 *     Ingest a vector of observations xs.
 *
 *   [chi, nu, regime, sigma] = cb_stream_mex('peek', ptr)
 *     Return current sufficient statistics and regime info.
 *     regime: 0=prior, 1=gamma, 2=CF, 3=ARS, 4=point mass
 *
 *   n = cb_stream_mex('rebuilds', ptr)
 *     Return ARS/common-hull rebuild count since stream creation.
 *
 *   [alpha, core_draws, rem_draws, rebuilds, active] =
 *       cb_stream_mex('core_stats', ptr)
 *     Return transported-core diagnostics.
 *
 *   cb_stream_mex('destroy', ptr)
 *     Free the stream.  Must be called when done.
 *
 * Usage pattern (called from CbStreamSampler.m):
 *   ptr = cb_stream_mex('create', chi, nu, seed, arm_idx, 256);
 *   theta = cb_stream_mex('draw', ptr, 1);
 *   cb_stream_mex('update', ptr, x_observed);
 *   cb_stream_mex('destroy', ptr);
 *
 * Build:
 *   mex -O CFLAGS="$CFLAGS -O3 -std=c99 -march=native" cb_stream_mex.c
 *   (Windows: mex -O COMPFLAGS="$COMPFLAGS /O2" cb_stream_mex.c)
 *
 * The C backend (cb_core.c) is included directly -- no separate compilation.
 *
 * MIT License.  Andrew Mullhaupt, Stony Brook University AMS/QF, 2026.
 */

#include "mex.h"
#include "matrix.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* Include the full C backend inline */
#include "cb_core.c"

/* =========================================================================
 * Helper: unpack cb_stream_t* from uint64 MATLAB scalar
 * ========================================================================= */
static cb_stream_t *unpack_ptr(const mxArray *arr)
{
    if (!mxIsUint64(arr) || mxGetNumberOfElements(arr) != 1)
        mexErrMsgIdAndTxt("cb_stream:badptr",
            "Stream handle must be a uint64 scalar.");
    uint64_t raw = *(uint64_t *)mxGetData(arr);
    if (raw == 0)
        mexErrMsgIdAndTxt("cb_stream:nullptr",
            "Stream handle is null (already destroyed?).");
    return (cb_stream_t *)(uintptr_t)raw;
}

/* =========================================================================
 * Helper: pack cb_stream_t* into uint64 MATLAB scalar
 * ========================================================================= */
static mxArray *pack_ptr(cb_stream_t *s)
{
    mxArray *out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *(uint64_t *)mxGetData(out) = (uint64_t)(uintptr_t)s;
    return out;
}

/* =========================================================================
 * MEX entry point
 * ========================================================================= */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || !mxIsChar(prhs[0]))
        mexErrMsgIdAndTxt("cb_stream:usage",
            "First argument must be a command string.");

    char cmd[32];
    mxGetString(prhs[0], cmd, sizeof(cmd));

    /* ------------------------------------------------------------------
     * CREATE
     * ptr = cb_stream_mex('create', chi, nu, base_seed, stream_idx, buf_cap)
     * ------------------------------------------------------------------ */
    if (strcmp(cmd, "create") == 0) {
        if (nrhs < 6)
            mexErrMsgIdAndTxt("cb_stream:create",
                "Usage: ptr = cb_stream_mex('create',chi,nu,seed,idx,buf_cap)");

        double   chi        = mxGetScalar(prhs[1]);
        double   nu         = mxGetScalar(prhs[2]);
        uint64_t base_seed  = mxIsUint64(prhs[3])
                              ? *(uint64_t *)mxGetData(prhs[3])
                              : (uint64_t)mxGetScalar(prhs[3]);
        uint64_t stream_idx = mxIsUint64(prhs[4])
                              ? *(uint64_t *)mxGetData(prhs[4])
                              : (uint64_t)mxGetScalar(prhs[4]);
        int      buf_cap    = (int)mxGetScalar(prhs[5]);

        if (!cb_stats_valid(chi, nu))
            mexErrMsgIdAndTxt("cb_stream:domain",
                "Require finite sufficient statistics 0 <= chi <= nu.");

        cb_stream_t *s = cb_stream_create(chi, nu, base_seed, stream_idx, buf_cap);
        if (!s)
            mexErrMsgIdAndTxt("cb_stream:alloc", "Failed to allocate stream.");

        /* MEMORY CONTRACT: the returned uint64 handle owns a heap-allocated
         * cb_stream_t.  The caller MUST eventually call 'destroy' on it.
         *
         * CbStreamSampler.m handles this automatically via its delete()
         * destructor -- MATLAB calls handle-class destructors before MEX
         * unload, so clear/clear all is safe when using the class wrapper.
         *
         * Direct MEX use (raw uint64 pointer) bypasses that safety net.
         * If the pointer is lost without calling 'destroy', the allocation
         * is orphaned until MATLAB restarts.  A mexAtExit registry would
         * fix this but requires maintaining a global stream table;
         * deferred in favour of the class wrapper as the public interface.
         *
         * Short version: use CbStreamSampler.  If you insist on raw
         * pointers, you own the cleanup. */
        plhs[0] = pack_ptr(s);
        return;
    }

    /* ------------------------------------------------------------------
     * DRAW
     * theta = cb_stream_mex('draw', ptr, n)
     * ------------------------------------------------------------------ */
    if (strcmp(cmd, "draw") == 0) {
        if (nrhs < 3)
            mexErrMsgIdAndTxt("cb_stream:draw",
                "Usage: theta = cb_stream_mex('draw', ptr, n)");

        cb_stream_t *s = unpack_ptr(prhs[1]);
        int n = (int)mxGetScalar(prhs[2]);
        if (n <= 0) {
            plhs[0] = mxCreateDoubleMatrix(0, 1, mxREAL);
            return;
        }

        plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
        double *out = mxGetPr(plhs[0]);
        if (cb_stream_draw(s, n, out) != n)
            mexErrMsgIdAndTxt("cb_stream:draw",
                "Backend failed to draw the requested sample count.");
        return;
    }

    /* ------------------------------------------------------------------
     * UPDATE (single observation)
     * cb_stream_mex('update', ptr, x)
     * ------------------------------------------------------------------ */
    if (strcmp(cmd, "update") == 0) {
        if (nrhs < 3)
            mexErrMsgIdAndTxt("cb_stream:update",
                "Usage: cb_stream_mex('update', ptr, x)");

        cb_stream_t *s = unpack_ptr(prhs[1]);
        double x = mxGetScalar(prhs[2]);
        if (!cb_stream_update(s, x))
            mexErrMsgIdAndTxt("cb_stream:domain",
                "Observation must be a finite value in [0,1].");
        return;
    }

    /* ------------------------------------------------------------------
     * UPDATE_BATCH (vector of observations)
     * cb_stream_mex('update_batch', ptr, xs)
     * ------------------------------------------------------------------ */
    if (strcmp(cmd, "update_batch") == 0) {
        if (nrhs < 3)
            mexErrMsgIdAndTxt("cb_stream:update_batch",
                "Usage: cb_stream_mex('update_batch', ptr, xs)");

        cb_stream_t *s  = unpack_ptr(prhs[1]);
        int          n  = (int)mxGetNumberOfElements(prhs[2]);
        const double *xs = mxGetPr(prhs[2]);
        if (!cb_stream_update_batch(s, xs, n))
            mexErrMsgIdAndTxt("cb_stream:domain",
                "Observations must be finite values in [0,1].");
        return;
    }

    /* ------------------------------------------------------------------
     * PEEK
     * [chi, nu, regime, sigma] = cb_stream_mex('peek', ptr)
     * ------------------------------------------------------------------ */
    if (strcmp(cmd, "peek") == 0) {
        if (nrhs < 2)
            mexErrMsgIdAndTxt("cb_stream:peek",
                "Usage: [chi,nu,regime,sigma] = cb_stream_mex('peek',ptr)");

        cb_stream_t *s = unpack_ptr(prhs[1]);
        double chi, nu, sigma;
        int regime;
        cb_stream_peek(s, &chi, &nu, &regime, &sigma);

        if (nlhs >= 1) plhs[0] = mxCreateDoubleScalar(chi);
        if (nlhs >= 2) plhs[1] = mxCreateDoubleScalar(nu);
        if (nlhs >= 3) plhs[2] = mxCreateDoubleScalar((double)regime);
        if (nlhs >= 4) plhs[3] = mxCreateDoubleScalar(sigma);
        return;
    }

    /* ------------------------------------------------------------------
     * REBUILDS
     * n = cb_stream_mex('rebuilds', ptr)
     * ------------------------------------------------------------------ */
    if (strcmp(cmd, "rebuilds") == 0) {
        if (nrhs < 2)
            mexErrMsgIdAndTxt("cb_stream:rebuilds",
                "Usage: n = cb_stream_mex('rebuilds', ptr)");

        cb_stream_t *s = unpack_ptr(prhs[1]);
        plhs[0] = mxCreateDoubleScalar((double)cb_stream_hull_rebuilds(s));
        return;
    }

    /* ------------------------------------------------------------------
     * CORE_STATS
     * [alpha, core_draws, rem_draws, rebuilds, active] =
     *     cb_stream_mex('core_stats', ptr)
     * ------------------------------------------------------------------ */
    if (strcmp(cmd, "core_stats") == 0) {
        if (nrhs < 2)
            mexErrMsgIdAndTxt("cb_stream:core_stats",
                "Usage: [alpha,core_draws,rem_draws,rebuilds,active] = cb_stream_mex('core_stats', ptr)");

        cb_stream_t *s = unpack_ptr(prhs[1]);
        double alpha;
        uint64_t core_draws, rem_draws, rebuilds, rem_cold, rem_hits;
        int active;
        cb_stream_core_stats(s, &alpha, &core_draws, &rem_draws, &rebuilds, &active,
                             &rem_cold, &rem_hits);

        if (nlhs >= 1) plhs[0] = mxCreateDoubleScalar(alpha);
        if (nlhs >= 2) plhs[1] = mxCreateDoubleScalar((double)core_draws);
        if (nlhs >= 3) plhs[2] = mxCreateDoubleScalar((double)rem_draws);
        if (nlhs >= 4) plhs[3] = mxCreateDoubleScalar((double)rebuilds);
        if (nlhs >= 5) plhs[4] = mxCreateDoubleScalar((double)active);
        if (nlhs >= 6) plhs[5] = mxCreateDoubleScalar((double)rem_cold);
        if (nlhs >= 7) plhs[6] = mxCreateDoubleScalar((double)rem_hits);
        return;
    }

    /* ------------------------------------------------------------------
     * DESTROY
     * cb_stream_mex('destroy', ptr)
     * ------------------------------------------------------------------ */
    if (strcmp(cmd, "destroy") == 0) {
        if (nrhs < 2)
            mexErrMsgIdAndTxt("cb_stream:destroy",
                "Usage: cb_stream_mex('destroy', ptr)");

        cb_stream_t *s = unpack_ptr(prhs[1]);
        cb_stream_destroy(s);
        /* Zero out the handle in-place if caller passed a variable
         * (not possible from MEX -- caller must null their own copy) */
        return;
    }

    mexErrMsgIdAndTxt("cb_stream:unknown",
        "Unknown command '%s'. Valid: create, draw, update, update_batch, peek, rebuilds, core_stats, destroy.",
        cmd);
}
