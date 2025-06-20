%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Underwater Image Enhancement Based on Adaptive Color Correction and Improved Retinex Algorithm
close all;
clear all;
clc;
warning off

%% Image input
rgbImage = imread('image11.jpg');
figure, imshow(rgbImage), title('original image');
rgbImage = im2double(rgbImage);
grayImage = rgb2gray(rgbImage);

%% Improved Retinex Algorithm
redChannel = rgbImage(:, :, 1);
greenChannel = rgbImage(:, :, 2);
blueChannel = rgbImage(:, :, 3);

meanR = mean2(redChannel);
meanG = mean2(greenChannel);
meanB = mean2(blueChannel);
meanGray = mean2(grayImage);

redChannel = (double(redChannel) * meanGray / meanR);
greenChannel = (double(greenChannel) * meanGray / meanG);
blueChannel = (double(blueChannel) * meanGray / meanB);

redChannel = redChannel - 0.3 * (meanG - meanR) .* greenChannel .* (1 - redChannel);
blueChannel = blueChannel + 0.3 * (meanG - meanB) .* greenChannel .* (1 - blueChannel);

rgbImage_white_balance = cat(3, redChannel, greenChannel, blueChannel);

figure('Name', 'Color Enhancement');
subplot(221); imshow(redChannel); title('Suppressed Red Channel');
subplot(222); imshow(blueChannel); title('Enhanced Blue Channel');
subplot(223); imshow(greenChannel); title('Green Channel');
subplot(224); imshow(rgbImage_white_balance); title('After White Balance');

%% Discrete Wavelet Transform (DWT) Processing
figure;
[LLr, LHr, HLr, HHr] = dwt2(rgbImage, 'haar');
subplot(221), imshow(mat2gray(LLr)), title('LOW LOW Data');
subplot(222), imshow(LHr), title('HIGH LOW Data');
subplot(223), imshow(HLr), title('LOW HIGH Data');
subplot(224), imshow(mat2gray(HHr)), title('HIGH HIGH Data');
out = idwt2(LLr, LHr, HLr, HHr, 'haar');
figure, imshow(mat2gray(out)); title('DWT Processed Image');

%% Retinex Processing with Gaussian Blur
img_filtered = rgb2gray(rgbImage_white_balance);
img_blur = imgaussfilt(img_filtered, 15);  % Gaussian blur
img_retinex = log(1 + img_filtered) - log(1 + img_blur);  % Retinex formula
img_retinex = imadjust(img_retinex, stretchlim(img_retinex), []);
figure, imshow(img_retinex); title('Retinex Enhanced Image');

%%% Optimal Weighted Fusion Retinex Process
normal_thr_limit = 0.5;
low_limt = 0.002;
up_limit = 0.999;

under_image = rgbImage;
[luminance, saliency, chromatic] = size(rgbImage);

if chromatic == 3
    inc_pixel_limit = 0.02;
    dec_pixel_limt = -0.02;
    max_chromatic = rgb2ntsc(under_image);
    mean_adjustment = inc_pixel_limit - mean(mean(max_chromatic(:, :, 2)));
    max_chromatic(:, :, 2) = max_chromatic(:, :, 2) + mean_adjustment * (0.256 - max_chromatic(:, :, 2));
    mean_adjustment = dec_pixel_limt - mean(mean(max_chromatic(:, :, 3)));
    max_chromatic(:, :, 3) = max_chromatic(:, :, 3) + mean_adjustment * (0.256 - max_chromatic(:, :, 3));
else
    max_chromatic = double(under_image) ./ 255;
end

mean_adjustment = normal_thr_limit - mean(mean(max_chromatic(:, :, 1)));
max_chromatic(:, :, 1) = max_chromatic(:, :, 1) + mean_adjustment * (.256 - max_chromatic(:, :, 1));
if chromatic == 3
    max_chromatic = ntsc2rgb(max_chromatic);
end

under_image = max_chromatic .* 255;

for k = 1:chromatic
    coeff = sort(reshape(under_image(:, :, k), luminance * saliency, 1));
    weight_min_pixel(k) = coeff(floor(low_limt * luminance * saliency));
    weight_max_pixel(k) = coeff(floor(up_limit * luminance * saliency));
end

if chromatic == 3
    weight_min_pixel = rgb2ntsc(weight_min_pixel);
    weight_max_pixel = rgb2ntsc(weight_max_pixel);
end

under_image = (under_image - weight_min_pixel(1)) / (weight_max_pixel(1) - weight_min_pixel(1));
figure, imshow(under_image); title('Fusion Retinex Process Image');

%% Temporal Correlation Pixels Separation
Adaptive_red = adapthisteq(under_image(:, :, 1),'clipLimit',0.01);
adapthisteq_green = adapthisteq(under_image(:, :, 2),'clipLimit',0.01);
adapthisteq_blue = adapthisteq(under_image(:, :, 3),'clipLimit',0.01);
enhanced_weight = cat(3, Adaptive_red, adapthisteq_green, adapthisteq_blue);
figure, imshow(enhanced_weight); % No mat2gray here

%% Sharpening
h = fspecial('unsharp',0.3);
sharpended = imfilter(enhanced_weight, h, 'replicate');
figure, imshow(sharpended); title('Sharpened Image');

%-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------%

%% MATLAB Code for Underwater Image Quality Metrics Integration
% Computes PCQI, UCIQE, UIQM, and IE for underwater images.

% Resize enhanced image to match the original size
enhanced = imresize(sharpended, [size(rgbImage,1), size(rgbImage,2)]);

% Compute Quality Metrics
pcqi_original = PCQI(rgbImage, rgbImage);
pcqi_enhanced = PCQI(rgbImage, enhanced);
uciqe_original = UCIQE(rgbImage);
uciqe_enhanced = UCIQE(enhanced);
uiqm_original = UIQM(rgbImage);
uiqm_enhanced = UIQM(enhanced);
ie_value = IE(rgbImage, enhanced);

% Display Results
fprintf('===================== Image Quality Metrics =====================\n');
fprintf('PCQI Value: %.4f\n', pcqi_enhanced);
fprintf('UCIQE Value: %.4f\n', uciqe_enhanced);
fprintf('UIQM Value: %.4f\n', uiqm_enhanced);
fprintf('IE Value: %.4f (Higher indicates more enhancement)\n', ie_value);
fprintf('================================================================\n');

%% Functions
% 1. PCQI (Perception-based Contrast Quality Index)
function pcqi_value = PCQI(original, enhanced)
    original_gray = double(rgb2gray(original));
    enhanced_gray = double(rgb2gray(enhanced));
    contrast_orig = stdfilt(original_gray, ones(3));
    contrast_enh = stdfilt(enhanced_gray, ones(3));
    luminance = mean(enhanced_gray(:)) / mean(original_gray(:));
    structure = corr2(original_gray, enhanced_gray);
    pcqi_value = luminance * structure * (mean(contrast_enh(:)) / mean(contrast_orig(:)));
end

% 2. UCIQE (Underwater Color Image Quality Evaluation)
function ucq = UCIQE(image)
    lab_image = rgb2lab(image);
    L = lab_image(:,:,1);
    a = lab_image(:,:,2);
    b = lab_image(:,:,3);
    sigma_c = std(sqrt(a.^2 + b.^2), 0, 'all');
    contrast_l = std(L(:));
    hsv_image = rgb2hsv(image);
    mu_s = mean(hsv_image(:,:,2), 'all');
    ucq = 0.4680 * sigma_c + 0.2745 * contrast_l + 0.2576 * mu_s;
end

% 3. UIQM (Underwater Image Quality Measure)
function uiq = UIQM(image)
    lab_image = rgb2lab(image);
    a = lab_image(:,:,2);
    b = lab_image(:,:,3);
    UICM = sqrt(var(a(:)) + var(b(:)));
    gray_image = rgb2gray(image);
    UISM = sum(sum(edge(gray_image, 'sobel')));
    UIConM = std2(gray_image);
    uiq = 0.0282 * UICM + 0.2953 * UISM + 3.5753 * UIConM;
end

% 4. IE (Image Enhancement Index)
function ie_value = IE(original, enhanced)
    original_gray = rgb2gray(original);
    enhanced_gray = rgb2gray(enhanced);
    diff = double(enhanced_gray) - double(original_gray);
    ie_value = sum(abs(diff(:))) / numel(original_gray);
end

% Compute percentage improvement
pcqi_improvement = ((pcqi_enhanced - pcqi_original) / pcqi_original) * 100;
uciqe_improvement = ((uciqe_enhanced - uciqe_original) / uciqe_original) * 100;
uiqm_improvement = ((uiqm_enhanced - uiqm_original) / uiqm_original) * 100;

% 1. Bar Chart - Comparing Quality Metrics
figure;
bar([pcqi_original, pcqi_enhanced; uciqe_original, uciqe_enhanced; uiqm_original, uiqm_enhanced; ie_value, ie_value]);
set(gca, 'XTickLabel', {'PCQI', 'UCIQE', 'UIQM', 'IE'});
legend('Original', 'Enhanced');
title('Comparison of Underwater Image Quality Metrics');
ylabel('Metric Value');

% 2. Line Graph - Percentage Improvement
figure;
plot(1:3, [pcqi_improvement, uciqe_improvement, uiqm_improvement], '-o', 'LineWidth', 2);
xticks(1:3);
xticklabels({'PCQI', 'UCIQE', 'UIQM'});
ylabel('Percentage Improvement (%)');
title('Quality Improvement After Enhancement');
grid on;

% 3. Histogram Comparisons
figure;
subplot(2,1,1);
imhist(rgb2gray(rgbImage));
title('Histogram of Original Image');
subplot(2,1,2);
imhist(rgb2gray(enhanced));
title('Histogram of Enhanced Image');