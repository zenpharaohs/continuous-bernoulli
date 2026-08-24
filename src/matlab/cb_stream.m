function s = cb_stream(chi, nu, varargin)
% CB_STREAM  Create a buffered streaming CB conjugate posterior sampler.
%
%   s = cb_stream(chi, nu)
%   s = cb_stream(chi, nu, 'seed', uint64(42), 'stream_idx', uint64(0), ...
%                          'buf_size', 256)
%
%   Returns a CbStreamSampler handle object.
%
%   Parameters:
%     chi         : initial sum of losses (chi = N_k * Z_bar_k)
%     nu          : initial pull count
%     seed        : uint64 base RNG seed (default: random)
%     stream_idx  : uint64 arm index for seed diversification (default: 0)
%     buf_size    : buffer capacity in samples (default: 256)
%
%   The stream object supports:
%     theta = s.draw(n)        -- draw n samples (serves from buffer)
%     s.update(x)              -- ingest one observation
%     s.update_batch(xs)       -- ingest a vector of observations
%     [chi,nu,regime,sigma] = s.peek()  -- current state
%     s.delete()               -- free C resources
%
%   Regime codes: 0=prior, 1=gamma, 2=CF, 3=ARS, 4=point mass
%
%   The buffer is pre-filled at construction and refilled automatically
%   on exhaustion.  On update(), the buffer is invalidated and rebuilt
%   with the new parameters at the next draw().
%
%   Seed discipline:
%     Each stream uses seed = base_seed XOR splitmix64(stream_idx+1).
%     For a K-arm bandit, use the same base_seed with stream_idx = 0..K-1
%     to get independent, non-overlapping RNG sequences per arm.
%
%   See also: CbStreamSampler, cb_sample, cb_stream_mex

parser = inputParser;
addRequired(parser, 'chi', @(x) isscalar(x) && isnumeric(x));
addRequired(parser, 'nu',  @(x) isscalar(x) && isnumeric(x));
addParameter(parser, 'seed',        uint64(randi(2^32)), @(x) isscalar(x));
addParameter(parser, 'stream_idx',  uint64(0),           @(x) isscalar(x));
addParameter(parser, 'buf_size',    256,                 @(x) isscalar(x) && x > 0);
parse(parser, chi, nu, varargin{:});
opts = parser.Results;

if ~isfinite(opts.chi) || ~isfinite(opts.nu) || ...
        opts.nu < 0 || opts.chi < 0 || opts.chi > opts.nu
    error('cb_stream:badStats', ...
        'Require finite sufficient statistics 0 <= chi <= nu.');
end

s = CbStreamSampler(double(opts.chi), double(opts.nu), ...
                    uint64(opts.seed), uint64(opts.stream_idx), ...
                    int32(opts.buf_size));
end
