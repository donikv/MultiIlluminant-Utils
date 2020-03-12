% example code to generate the single illuminant results in TABLE I and II of publication

real_world_flag=0;   % real_world_flag=1 for real-world dataset / real_world_flag=0 laboratory dataset

method = 1;
% choose method
% method = 1 -> Do Nothing
% method = 2 -> Grey-World
% method = 3 -> White-Patch
% method = 4 -> Grey-Edge
% method = 5 -> 2nd order Grey-Edge

addpath './generalized_gray_world/'
if(real_world_flag)
    load('image_names_real');
    pathImages='./../realworld/img/';
    pathGT='./../realworld/groundtruth/';
    pathMasks='./../realworld/masks/';
else
    load('image_names_lab');
    pathImages='./../lab/img/';
    pathGT='./../lab/groundtruth/';
    pathMasks='./../lab/masks/';
end

nIm=size(image_names,2);  % number of images
errors=zeros(nIm,1);

for ii=1:nIm  % loop over images;
    input_im = double(imread([pathImages,image_names{ii},'.png']));
    GT_im = double(imread([pathGT,image_names{ii},'.png']));
    mask = double(imread([pathMasks,image_names{ii},'.png']));
    switch method
        case 1
            white_R=1; white_G=1; white_B=1;  % Do Nothing (DN)
        case 2
            [white_R ,white_G ,white_B,output_data] = jvdw_general_cc(input_im,0,1,0);   % Grey-World (GW)
        case 3
            [white_R ,white_G ,white_B,output_data] = jvdw_general_cc(input_im,0,-1,0); % white-patch (WP)
        case 4
            [white_R ,white_G ,white_B,output_data] = jvdw_general_cc(input_im,1,1,1);  % Grey-Edge (GE1)
        case 5
            [white_R ,white_G ,white_B,output_data] = jvdw_general_cc(input_im,2,1,1);  % second order GE (GE2)
    end
    EstIl = repmat(reshape([white_R,white_G,white_B],1,1,3),size(GT_im,1),size(GT_im,2));
    adist=angDistPixelwise(GT_im.*repmat(mask,[1,1,3]),EstIl);
    errors(ii)=mean(adist)/pi*180;          % error in degrees
end
fprintf(1,'The mean error=%f\n',mean(errors));
fprintf(1,'The median error=%f\n',median(errors));
