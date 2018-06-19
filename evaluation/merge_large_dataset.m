function merge_large_trainset(data_dir)
	files = dir(sprintf('%s/posterior_param_*.mat',data_dir));
        file_names = sort({files.name});
        encoded = struct;
        encoded.structure_param = [];
	for i=1:numel(file_names)
             file = file_names{i};
             fprintf(file);
             tmp = load(sprintf('%s/%s',data_dir,file));
             encoded.structure_param = [encoded.structure_param; tmp.encoded.structure_param];
        end	
        save(sprintf('%s/posterior_param.mat',data_dir), 'encoded');
end
