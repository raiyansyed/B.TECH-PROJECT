%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Underwater Image Enhancement Based on Adaptive Color Correction and Improved Retinex Algorithm
close all;
clear all;
clc;
warning off
%% Image input
% We take a RGB image as input and convert it to grayscale and store it in
% another variable, so we can get the mean luminance.

rgbImage=imread('image11.jpg');
figure,imshow(rgbImage),title('original image');
rgbImage=im2double(rgbImage);
grayImage = rgb2gray(rgbImage); 
%% Improved Retinex Algorithm
% Extract the individual red, green, and blue color channels.
redChannel = rgbImage(:, :, 1);
greenChannel = rgbImage(:, :, 2);
blueChannel = rgbImage(:, :, 3);

meanR = mean2(redChannel);
meanG = mean2(greenChannel);
meanB = mean2(blueChannel);
meanGray = mean2(grayImage);

% Make all channels have the same mean
redChannel = (double(redChannel) * meanGray / meanR);
greenChannel = (double(greenChannel) * meanGray / meanG);
blueChannel = (double(blueChannel) * meanGray / meanB);

%redChannel and blueChannel Correction
redChannel=redChannel-0.3*(meanG-meanR).*greenChannel.*(1-redChannel);
blueChannel=blueChannel+0.3*(meanG-meanB).*greenChannel.*(1-blueChannel);


% Recombine separate color channels into a single, true color RGB image.
rgbImage_white_balance = cat(3, redChannel, greenChannel, blueChannel);

figure('Name','Color Enhancement');
subplot(221);
imshow(redChannel);
title('Suppressed Red Channel');

subplot(222);
imshow(blueChannel);
title('Enhanced Blue Channel');

subplot(223);
imshow(greenChannel);
title('Green Channel');

subplot(224);
imshow(rgbImage_white_balance);
title('After White balance');

figure,
[LLr,LHr,HLr,HHr]=dwt2(rgbImage,'haar');
subplot(221),imshow(mat2gray(LLr)),title('LOW LOW Data');
subplot(222),imshow(LHr),title('HIGH LOW Data');
subplot(223),imshow(HLr),title('LOW HIGH Data');
subplot(224),imshow(mat2gray(HHr)),title('HIGH HIGH Data');
out=idwt2(LLr,LHr,HLr,HHr,'haar');
figure,imshow(mat2gray(out));title('dwt process image')
%%% optimal weighted fusion  Retinex process
normal_thr_limit=0.5;
low_limt=0.002;
up_limit=0.999;
%----------------------------------------------------------------------
under_image=rgbImage;
[luminance saliency chromatic]=size(rgbImage);
%----------------------------------------------------------------------
if chromatic==3
    inc_pixel_limit=0.02;dec_pixel_limt=-0.02;
    max_chromatic=rgb2ntsc(under_image);
    mean_adjustment=inc_pixel_limit-mean(mean(max_chromatic(:,:,2)));
    max_chromatic(:,:,2)=max_chromatic(:,:,2)+mean_adjustment*(0.256-max_chromatic(:,:,2));
    mean_adjustment=dec_pixel_limt-mean(mean(max_chromatic(:,:,3)));
    max_chromatic(:,:,3)=max_chromatic(:,:,3)+mean_adjustment*(0.256-max_chromatic(:,:,3));
else
    max_chromatic=double(under_image)./255;
end
%----------------------------------------------------------------------
mean_adjustment=normal_thr_limit-mean(mean(max_chromatic(:,:,1)));
max_chromatic(:,:,1)=max_chromatic(:,:,1)+mean_adjustment*(.256-max_chromatic(:,:,1));
if chromatic==3
    max_chromatic=ntsc2rgb(max_chromatic);
end
%----------------------------------------------------------------------
under_image=max_chromatic.*255;
%--------------------caliculate the max to min pixels----------------------
for k=1:chromatic
    coeff=sort(reshape(under_image(:,:,k),luminance*saliency,1));
    weight_min_pixel(k)=coeff(floor(low_limt*luminance*saliency));
    weight_max_pixel(k)=coeff(floor(up_limit*luminance*saliency));
end
%----------------------------------------------------------------------
if chromatic==3
    weight_min_pixel=rgb2ntsc(weight_min_pixel);
    weight_max_pixel=rgb2ntsc(weight_max_pixel);
end
%----------------------------------------------------------------------
under_image=(under_image-weight_min_pixel(1))/(weight_max_pixel(1)-weight_min_pixel(1));
figure,imshow(under_image);title('fusion  Retinex process image');

    
% % %%%%%%%%%%%%%%%temporal correlation pixels separation
Adaptive_red = adapthisteq(under_image(:,:,1));
            adapthisteq_green = adapthisteq(under_image(:,:,2));
            adapthisteq_blue = adapthisteq(under_image(:,:,3));
            enhanced_weight = cat(3,Adaptive_red,adapthisteq_green,adapthisteq_blue);
            figure,imshow(mat2gray(enhanced_weight));title('proposed enhanced output image');
% crop paerticular part
cropping=imcrop(enhanced_weight);
figure,imshow(cropping),title('selected part');
h=fspecial('unsharp');
sharpended=imfilter(cropping,h,'replicate');
figure,imshow(sharpended);title('sharpended image');


%     %% metric values
%     disp('metric values for enhance image')
%     [psnr,mse,NC,ssim] = measerr(rgbImage,enhanced_weight);
%   
%      fprintf('\npsnr: %1f ', psnr(:,:,1));
%  fprintf('\nmse: %1f ', mse(:,:,1));
%       fprintf('\n Normalized Co-efficent: %1f ', NC(:,:,1));
%  fprintf('\n SSIM: %1f ', ssim(:,:,1));
%    fprintf('\n entropy: %1f ', entropy(enhanced_weight));
