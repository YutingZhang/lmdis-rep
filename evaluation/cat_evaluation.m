function kp_evaluation(train_pred_file, train_gt_file, test_pred_file, test_gt_file)
        fprintf('Load the predicted keypoints and ground truth keypoints for the training dataset\n');
        train_pred = load(train_pred_file);
	train_pred_kp = train_pred.encoded.structure_param;
        train_gt = load(train_gt_file);
	train_gt_hw = train_gt.hw*80/100;
        train_gt_kp = train_gt.gt(:,[1:4,6:7,9],:)-10;
        train_gt_kp(:,:,1) = train_gt_kp(:,:,1)./double(repmat(squeeze(train_gt_hw(:,1)),[1, size(train_gt_kp,2)]));
        train_gt_kp(:,:,2) = train_gt_kp(:,:,2)./double(repmat(squeeze(train_gt_hw(:,2)),[1, size(train_gt_kp,2)]));
        test_pred = load(test_pred_file);
        test_pred_kp = test_pred.encoded.structure_param;
        test_gt = load(test_gt_file);
        test_gt_kp = test_gt.gt(:,[1:4,6:7,9],:)-10;
        test_gt_hw = test_gt.hw*80/100;
        test_gt_kp(:,:,1) = test_gt_kp(:,:,1)./double(repmat(squeeze(test_gt_hw(:,1)),[1, size(test_gt_kp,2)]));
        test_gt_kp(:,:,2) = test_gt_kp(:,:,2)./double(repmat(squeeze(test_gt_hw(:,2)),[1, size(test_gt_kp,2)]));

        train_gt_kp = train_gt_kp - 0.5;
        test_gt_kp = test_gt_kp - 0.5;
        train_pred_kp = train_pred_kp - 0.5;
        test_pred_kp = test_pred_kp - 0.5;
       
        W = linear_regressor(train_pred_kp, train_gt_kp);
        test_pred_kp = reshape(test_pred_kp,[size(test_pred_kp,1), size(test_pred_kp,2)*size(test_pred_kp,3)] );
        test_fit_kp = test_pred_kp*W;
        test_fit_kp = reshape(test_fit_kp, [size(test_pred_kp,1), size(test_fit_kp,2)/2, 2]);
        mean_error = mean_error_IOD(test_fit_kp, test_gt_kp);
        mean_error
end
