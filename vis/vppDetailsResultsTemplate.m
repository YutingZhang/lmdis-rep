function vppDetailsResultsTemplate( ...
    show_func, result_path, step, samples_ids, save_to_file, output_pattern)

if ~exist('save_to_file', 'var') || isempty(save_to_file)
    save_to_file = false;
end

if ~iscell(result_path)
    result_path = {result_path};
end

if ~exist('output_pattern', 'var') || isempty(output_pattern)
    output_pattern = '%s/%d';
end

auto_continue = false;
for k = 1:numel(result_path)
    is_last_single = (k==numel(result_path));
    auto_continue = vppDetailsResultsTemplate_Single( ...
        show_func, result_path{k}, step, samples_ids, save_to_file, ...
        auto_continue, is_last_single, output_pattern);
end


function batch_mode = vppDetailsResultsTemplate_Single( ...
    show_func, result_path, step, samples_ids, save_to_file, ...
    auto_continue, is_last_single, output_pattern)

batch_mode = auto_continue;
result_path0 = result_path;

if isempty(step)
    step = 'latest';
end
step0 = step;
if ischar(step0) && (strcmp(step0, 'all') || strcmp(step0, 'latest'))
    all_available_steps = dir(fullfile(result_path, 'test.snapshot/step_*'));
    all_available_steps = {all_available_steps.name};
    all_available_steps = ...
        cellfun(@(a) str2double(a(6:end)), all_available_steps);
    all_available_steps(isnan(all_available_steps)) = [];
    all_available_steps = sort(all_available_steps);
    step = all_available_steps;
    if exist(fullfile(result_path, 'test.final'), 'file')
        step(end+1) = -1;
    end
    if isempty(step)
        fprintf(2, 'No test snapshot found: %s\n', result_path)
        return
    end
    if strcmp(step0, 'latest')
        step = step(end);
    end
end

if save_to_file
    callback = figure_show_callback(output_pattern, auto_continue);
else
    callback = figure_show_callback();
end

if ischar(step)
    step = {step};
end

for k = 1:numel(step)
    if iscell(step(k))
        step_k = step{k};
    else
        step_k = step(k);
    end
    if step_k<0
        step_k = 'final';
    end
    if ischar(step_k)
        step_result_path = fullfile(result_path, sprintf('test.%s', step_k));
        step_str = ['step ' step_k];
    else
        step_result_path = fullfile(result_path, sprintf('test.snapshot/step_%d', step_k));
        step_str = sprintf('step %d', step_k);
    end
    if ~exist(step_result_path, 'dir')
        fprintf(2, 'Test snapshot folder not available: %s\n', step_result_path)
        continue;
    end
    fprintf('Display: %s\n', step_result_path)
    % step_title = sprintf('step %d', step(k));
    step_title = sprintf('%s\n%s ', ...
        strrep(wrap_str_at_comma(result_path0, 100), '_', '\_'), step_str);
    show_func( ...
        step_result_path, step_title, samples_ids, callback);
    if k<numel(step) || ~is_last_single
        callback.callback_dummy()
    end
end

batch_mode = callback.auto_continue;


function r = wrap_str_at_comma(s, line_length)

r = '';
while length(s)>line_length
    bpb = s==',';
    bploc = find(bpb(1:min(end,line_length)),1,'last');
    if isempty(bploc)
        bploc = find(bpb,1);
    end
    if ~isempty(bploc)
        r = [r, newline, s(1:bploc)];
        s = s(bploc+1:end);
    end
end
r = [r, newline, s];
r = strtrim(r);


