%% build_cb_stream.m
% Build the cb_stream_mex MEX file.
% Run from the repository root directory.

cb_root  = pwd;
src_c    = fullfile(cb_root, 'src', 'c');
src_m    = fullfile(cb_root, 'src', 'matlab');
mex_src  = fullfile(src_c, 'cb_stream_mex.c');

if ~exist(mex_src, 'file')
    error('build_cb_stream:notfound', ...
        'Cannot find %s\nRun from the repository root directory.', mex_src);
end

% Add src/matlab to path so cb_stream.m and friends are available
% immediately after build (also survives session crashes).
addpath(cb_root);   % for cb_validate, cb_qualify, etc.
addpath(src_m);     % for cb_stream, cb_mode, bft_*, etc.

fprintf('Building cb_stream_mex...\n');

% Windows installs that use MinGW may need MW_MINGW64_LOC. Leave other
% platforms alone so macOS/Linux builds can use their selected compiler.
if ispc && isempty(getenv('MW_MINGW64_LOC'))
    mingw_root = 'C:\ProgramData\MATLAB\SupportPackages\R2025b\3P.instrset\mingw_w64.instrset';
    if exist(mingw_root, 'dir')
        setenv('MW_MINGW64_LOC', mingw_root);
    end
end

mex('-O', ['-I' src_c], '-outdir', cb_root, mex_src);
fprintf('Build complete: %s\n', fullfile(cb_root, ['cb_stream_mex.' mexext]));
