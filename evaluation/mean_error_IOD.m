function mean_error = mean_error_IOD(fit_kp, gt_kp)
    %assume the order of keypoints are  right eye, left_eye, nose,  mouth_corner_r mouth_corner_l  
    fprintf('Calculate the error for each face \n');
    size(gt_kp)
    error_list = [];
    for i=1:size(gt_kp, 1)
        fit_keypoints = squeeze(fit_kp(i, :, :));
        gt_keypoints = squeeze(gt_kp(i, :, :));
        face_error = 0;
        for k = 1:size(gt_kp, 2)
            face_error = face_error + norm(fit_keypoints(k,:)-gt_keypoints(k,:));
        end
        face_error = face_error/size(gt_kp, 2);
        if size(gt_keypoints,1)==5
            right_pupil = gt_keypoints(1, :);
            left_pupil = gt_keypoints(2, :);
        elseif size(gt_keypoints,1)==68
            left_pupil = gt_keypoints(8, :);
            right_pupil = gt_keypoints(11, :);
        elseif (size(gt_keypoints,1)==9 || size(gt_keypoints,1)==7)
            left_pupil = gt_keypoints(1, :);
            right_pupil = gt_keypoints(2, :);
        elseif (size(gt_keypoints,1)==6)
            left_pupil = gt_keypoints(1, :);
            right_pupil = gt_keypoints(2, :);
        elseif (size(gt_keypoints, 1)==32)
            left_pupil = [0,0];
            right_pupil = [0,1];
        end
        IOD = norm(right_pupil-left_pupil);
        if(IOD~=0)
            face_error_normalized = face_error/IOD;
            error_list = [error_list, face_error_normalized];
        end 
    end
    mean_error = mean(error_list);
end
