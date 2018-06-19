function [W] = linear_regressor(pred_kp, gt_kp)
   fprintf('clean the gt keypoints mat and prediction keypoints mat \n');
   nan_rows = [];
   for k=1:size(gt_kp, 1)
       if (max(gt_kp(k,:))==0)
           nan_rows = [nan_rows, k];
       end
   end
   if length(nan_rows) > 0
       fprintf('nan_rows/n');
       nan_rows
   end
   for i = size(nan_rows, 2):-1:1
       row_idx = nan_rows(1,i);
       gt_kp = gt_kp([1:(row_idx-1),(row_idx+1):size(gt_kp,1)], :, :);
       pred_kp = pred_kp([1:(row_idx-1),(row_idx+1):size(pred_kp,1)], :, :);
   end
   clean_gt_kp = gt_kp;
   X = reshape(pred_kp,[size(pred_kp,1), size(pred_kp,2)*size(pred_kp,3)] );
   Y = reshape(clean_gt_kp,[size(clean_gt_kp,1), size(clean_gt_kp,2)*size(clean_gt_kp,3)] );
   %The most simple linear regression WX = Y, W = X\Y
   W = X\Y;
end
