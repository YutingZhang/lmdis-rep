function vppAutoKeypointImageRecon(result_path, step, samples_ids, save_to_file, type_ids)

if ~exist('type_ids', 'var')
    type_ids = {};
end

if ischar(type_ids)
    type_ids = {type_ids};
end

vppDetailsResultsTemplate( ...
    @(varargin) vppAutoKeypointReconPrior_Internal(type_ids, varargin{:}), ...
    result_path, step, samples_ids, save_to_file, '%s/%s_%s');

function vppAutoKeypointReconPrior_Internal( ...
    type_ids, result_path, the_title, samples_ids, callback)

A = load(fullfile(result_path, 'posterior_param.mat'));

if ~isfield(A.encoded, 'structure_param')
    A.encoded.structure_param = zeros(numel(samples_ids),0);
end

if ~isfield(A.decoded, 'structure_param')
    A.decoded.structure_param = zeros(numel(samples_ids),0);
end


h = size(A.data, 2);
w = size(A.data, 3);
sample_num = size(A.data, 1);

if isempty(samples_ids)
    samples_ids = 1:sample_num;
end
samples_ids = reshape(samples_ids, 1, numel(samples_ids));

N = numel(samples_ids);
fw = ceil(sqrt(N*(4/3)*(h/w)));
if (N/fw)==ceil(N/fw)
    ;
elseif (N/(fw-1))==ceil((N-1)/(fw-1))
    fw = fw-1;
elseif (N/(fw+1))==ceil((N-1)/(fw+1))
    fw = fw+1;
end
fh = ceil(N/fw);

id_str = sprintf('%d-%d', min(samples_ids), max(samples_ids));

type_id_list = {'data-encoded', 'data-decoded', 'recon-encoded', 'recon-decoded'};

if ~isempty(type_ids)
    assert(all(ismember(type_ids, type_id_list)), 'unrecognized ids');
    type_id_list = type_ids;
end

for type_id = 1:length(type_id_list)

    type_str = type_id_list{type_id};
    
    D = struct();
    switch type_str
        case 'data-encoded'
           D.vis = A.data;
           D.structure_param = A.encoded.structure_param;
        case 'data-decoded'
           D.vis = A.data;
           D.structure_param = A.decoded.structure_param;
        case 'recon-encoded'
           D.vis = A.decoded.vis;
           D.structure_param = A.encoded.structure_param;
        case 'recon-decoded'
           D.vis = A.decoded.vis;
           D.structure_param = A.decoded.structure_param;
        otherwise
            error('Internal error: Unrecognized type')
    end
    
    figure(1)
    set(gcf, 'color', 'white');
    clf
    for j = 1:N
        sidx = samples_ids(j);
        subplot(fh, fw, j)
        vppAutoKeypointShowSingle(squeeze(D.vis(sidx, :, :, :)), squeeze(D.structure_param(sidx, :, :)))
        title(int2str(sidx))

        ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
        text(0.5, 1,sprintf('%s %s (samples %s) %s', '\bf', the_title, id_str, type_str), ...
            'HorizontalAlignment' ,'center','VerticalAlignment', 'top');

    end
    
    if type_id ~= length(type_id_list)
        callback.callback( ...
            [result_path '_recon_keypoints_batch'], ...
            id_str, type_str ...
        );
    else
        callback.callback_no_user_input( ...
            [result_path '_recon_keypoints_batch'], ...
            id_str, type_str ...
        );
    end
end

