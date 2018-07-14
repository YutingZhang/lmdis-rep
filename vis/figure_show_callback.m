classdef figure_show_callback < handle

    properties (GetAccess=public, SetAccess=protected)
        output_path_pattern = []
        auto_continue = false
    end
    
    methods (Access=public)
        
        function obj = figure_show_callback(output_path_pattern, auto_continue)
            if nargin < 1
                output_path_pattern = [];
            end
            if nargin < 2
                auto_continue = false;
            end
            obj.output_path_pattern = output_path_pattern;
            if isempty(output_path_pattern)
                assert(~auto_continue, 'Cannot use auto continue for display mode');
            end
            obj.auto_continue = auto_continue;
        end
        
        function callback(obj, varargin)
            obj.callback_dummy()
            obj.callback_no_user_input(varargin{:})
        end
        
        function callback_dummy(obj)
            if ~obj.auto_continue
                c = waitforbuttonpress;
                if ~isempty(obj.output_path_pattern) && c
                    obj.auto_continue = true;
                    fprintf('Batch mode started\n')
                end
            end
        end
        
        function callback_no_user_input(obj, varargin)
            if ~isempty(obj.output_path_pattern)
                the_figure = gcf;
                the_axis = gca;
                file_path = sprintf(obj.output_path_pattern, varargin{:});
                file_path = [file_path '.eps'];
                [parent_folder, ~, ~] = fileparts(file_path);
                if ~exist(parent_folder, 'file')
                    mkdir(parent_folder)
                end
                fprintf('%s : ', file_path); tic
                saveas(the_figure, file_path, 'epsc');
                toc
                figure(the_figure)
                axes(the_axis)
                pause(0.00001)
            end
        end
        
    end
    
end
