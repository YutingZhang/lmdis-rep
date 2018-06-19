function vppAutoKeypointShow(im, keypoints, output_dir)

output_to_dir = exist('output_dir', 'var') && ~isempty(output_dir);
if output_to_dir
    if ~exist(output_dir, 'dir')
        mkdir(output_dir)
    end
end

batch_mode_started = false;
for k=1:size(im, 1)
    I = squeeze(im(k, :, :, :));
    clf
    the_figure = gcf;
    P=double(squeeze(keypoints(k,:,:)));
    vppAutoKeypointShowSingle(I, P)
    set(gca,'visible','off');
    set(gcf,'color','white');
    if output_to_dir
        fprintf('%d / %d\n', k, size(im, 1))
        saveas(the_figure, fullfile(output_dir, sprintf('%d.eps', k)), 'epsc')
    else
        while waitforbuttonpress; end
    end
end
