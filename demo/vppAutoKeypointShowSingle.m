function vppAutoKeypointShowSingle(I, P)

colormap(gca, 'gray')
imagesc(I)
pbaspect(gca, [size(I,2), size(I,1), 1])
hold on
if isempty(P)
    hold off;
    return
end
H=size(I,1);
W=size(I,2);
ASPECT_RATIO=W/H;
asrq=sqrt(ASPECT_RATIO);

P(:,[1,2]) = bsxfun(@times, P(:,[1,2]), [size(I,1)*asrq, size(I,2)/asrq])+1;
P = double(P);
if 0
    for j = 1:size(P,1)
        the_color = [1 0 0];
        plot(P(j,2),P(j,1), ...
            'o', 'Color', the_color, 'MarkerFaceColor', the_color);
        text(P(j,2),P(j,1), int2str(j), ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', ...
            'FontSize', 12, ...
            'Color', the_color ...
        );
    end
else
    %the_color_list = linspecer(size(P,1));
    the_color_list = jet(size(P,1));
    for j = 1:size(P,1)
        the_color = the_color_list(j,:);
        plot(P(j,2),P(j,1), ...
            '+', 'Color', the_color, 'LineWidth', 5, 'MarkerSize', 8);
    end
end

hold off
