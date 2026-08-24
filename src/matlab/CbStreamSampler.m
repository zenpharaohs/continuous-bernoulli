classdef CbStreamSampler < handle
% CbStreamSampler  Buffered streaming CB conjugate posterior sampler.
%
%   Wraps the C backend (cb_stream_mex) for efficient sequential sampling
%   from the CB posterior p(theta | chi, nu).
%
%   Construct via cb_stream():
%     s = cb_stream(chi, nu)
%     s = cb_stream(chi, nu, 'seed', uint64(42), 'stream_idx', uint64(k))
%
%   Methods:
%     theta = s.draw(n)          draw n samples; refills buffer as needed
%     s.update(x)                ingest one observation x in [0,1]
%     s.update_batch(xs)         ingest a vector of observations
%     [chi,nu,r,sig] = s.peek()  current chi, nu, regime (0-4), sigma
%     n = s.rebuilds()           ARS/common-hull rebuild count
%     stats = s.core_stats()     transported-core diagnostics
%     s.stats()                  print usage statistics
%     s.delete()                 free C resources (called automatically by GC)
%
%   Regime codes: 0=prior, 1=Gamma, 2=CF, 3=ARS, 4=point mass
%
%   For a K-arm Thompson bandit, create K streams with the same base_seed
%   and stream_idx = 0..K-1:
%     for k = 1:K
%         arms{k} = cb_stream(chi0(k), nu0(k), 'stream_idx', uint64(k-1));
%     end
%   Each arm then has an independent, reproducible RNG sequence.
%
%   See also: cb_stream, cb_sample, cb_stream_mex

    properties (Access = private)
        ptr_          % uint64: C-side cb_stream_t pointer
        buf_size_     % buffer capacity

        % Stats
        n_draws_      % total samples drawn
        n_updates_    % total update() calls
        n_refills_    % tracked refills (approximate)
    end

    methods

        function obj = CbStreamSampler(chi, nu, seed, stream_idx, buf_size)
        % Constructor: called by cb_stream().  Do not call directly.
            if nargin < 5 || buf_size <= 0, buf_size = int32(256); end
            obj.ptr_       = cb_stream_mex('create', ...
                                 double(chi), double(nu), ...
                                 uint64(seed), uint64(stream_idx), ...
                                 int32(buf_size));
            obj.buf_size_  = double(buf_size);
            obj.n_draws_   = 0;
            obj.n_updates_ = 0;
            obj.n_refills_ = 0;
        end

        function theta = draw(obj, n)
        % DRAW  Draw n samples from the current posterior.
        %   theta = s.draw(n)
        %   theta = s.draw()     % draw 1 sample
            if nargin < 2, n = 1; end
            theta = cb_stream_mex('draw', obj.ptr_, int32(n));
            obj.n_draws_ = obj.n_draws_ + n;
        end

        function update(obj, x)
        % UPDATE  Ingest one new observation x in [0,1].
        %   s.update(x)
        %   Updates chi += x, nu += 1, invalidates buffer, re-determines regime.
            cb_stream_mex('update', obj.ptr_, double(x));
            obj.n_updates_ = obj.n_updates_ + 1;
        end

        function update_batch(obj, xs)
        % UPDATE_BATCH  Ingest a vector of observations xs.
        %   s.update_batch(xs)
            cb_stream_mex('update_batch', obj.ptr_, double(xs(:)));
            obj.n_updates_ = obj.n_updates_ + numel(xs);
        end

        function [chi, nu, regime, sigma] = peek(obj)
        % PEEK  Return current sufficient statistics and regime.
        %   [chi, nu, regime, sigma] = s.peek()
        %   regime: 0=prior, 1=Gamma, 2=CF, 3=ARS, 4=point mass
            [chi, nu, regime, sigma] = cb_stream_mex('peek', obj.ptr_);
        end

        function n = rebuilds(obj)
        % REBUILDS  Return ARS/common-hull rebuild count since construction.
            n = cb_stream_mex('rebuilds', obj.ptr_);
        end

        function st = core_stats(obj)
        % CORE_STATS  Return transported-core diagnostics.
            [alpha, core_draws, rem_draws, rebuilds, active, rem_cold, rem_hits] = ...
                cb_stream_mex('core_stats', obj.ptr_);
            st = struct('alpha_hat', alpha, ...
                        'core_draws', core_draws, ...
                        'remainder_draws', rem_draws, ...
                        'rebuilds', rebuilds, ...
                        'core_active', active, ...
                        'remainder_cold_builds', rem_cold, ...
                        'remainder_cache_hits', rem_hits);
        end

        function stats(obj)
        % STATS  Print usage summary.
            [chi, nu, regime, sigma] = obj.peek();
            regime_names = {'prior','gamma','CF','ARS','point'};
            r_idx = min(max(regime + 1, 1), 5);
            fprintf('CbStreamSampler stats:\n')
            fprintf('  chi=%.4g  nu=%.4g  Z_bar=%.4f\n', chi, nu, chi/max(nu,1))
            fprintf('  regime=%s  sigma=%.4f\n', regime_names{r_idx}, sigma)
            fprintf('  total draws:   %d\n', obj.n_draws_)
            fprintf('  total updates: %d\n', obj.n_updates_)
            fprintf('  buffer size:   %d\n', obj.buf_size_)
            if obj.n_draws_ > 0 && obj.buf_size_ > 0
                fprintf('  approx refills: %d\n', ...
                    ceil(obj.n_draws_ / obj.buf_size_))
            end
        end

        function delete(obj)
        % DELETE  Free C resources.  Called automatically by MATLAB GC.
            if ~isempty(obj.ptr_) && obj.ptr_ ~= uint64(0)
                cb_stream_mex('destroy', obj.ptr_);
                obj.ptr_ = uint64(0);
            end
        end

    end % methods

end % classdef
