% Focused MATLAB/MEX endpoint regression. Run build_cb_stream first.
repo_root = fileparts(fileparts(mfilename('fullpath')));
addpath(repo_root, fullfile(repo_root,'src','matlab'));
endpoint_results = cb_validate_endpoints(2000);
assert(numel(endpoint_results)==66 && all([endpoint_results.passed]));
assert(sum(strcmp({endpoint_results.path},'live core buffer'))==2);
fprintf('MATLAB/MEX endpoint regression PASS\n');
