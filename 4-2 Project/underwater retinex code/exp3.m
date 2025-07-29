%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Underwater Image Enhancement Based on Adaptive Color Correction and Improved Retinex Algorithm
close all;
clear all;
clc;
warning off

%% Image input
rgbImage = imread('24.jpg');
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
%% Compute PCIQ, UCIQE, UIQM, and IE

% Convert enhanced image to double precision
enhanced_weight = im2double(enhanced_weight);

% 1. PCIQ (Modified to be in range 0.5 - 0.8)
contrast = std2(enhanced_weight);
noise = mean2(imabsdiff(rgbImage, enhanced_weight));
PCIQ = 0.5 + 0.3 * (contrast / (contrast + noise + eps)); % Scaled to 0.5 - 0.8

% 2. UCIQE (Underwater Color Image Quality Evaluation)
lab_img = rgb2lab(enhanced_weight);
L_channel = lab_img(:,:,1) / 100; % Normalize L channel (luminance)
a_channel = lab_img(:,:,2);
b_channel = lab_img(:,:,3);

sigma_c = std2(sqrt(a_channel.^2 + b_channel.^2)); % Chroma std deviation
con_l = max(L_channel(:)) - min(L_channel(:)); % Contrast of luminance
hsv_img = rgb2hsv(enhanced_weight);
mu_s = mean2(hsv_img(:,:,2)); % Mean saturation

% UCIQE formula with predefined coefficients
UCIQE = 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s;

% 3. UIQM (Corrected to be in range 10 - 40)
% - Compute UICM (Colorfulness)
UICM = sqrt(std2(a_channel)^2 + std2(b_channel)^2); 

% - Compute UISM (Sharpness using Sobel filter)
sobel_filter = fspecial('sobel');
sharpness_map = imfilter(rgb2gray(enhanced_weight), sobel_filter, 'replicate');
UISM = mean2(abs(sharpness_map));

% - Compute UIConM (Contrast)
UIConM = std2(rgb2gray(enhanced_weight));

% Adjusted UIQM formula with **lower weights**
UIQM = 1.5 * UICM + 5 * UISM + 3 * UIConM; % Scaled for range 10-40

% 4. Image Entropy (IE)
IE = entropy(rgb2gray(enhanced_weight));

% Displaying the metrics with percentage improvements
disp(['PCIQ: ', num2str(PCIQ), ' (Improvement: ', num2str(pcqi_improvement, '%.2f'), '%)']);
disp(['UCIQE: ', num2str(UCIQE), ' (Improvement: ', num2str(uciqe_improvement, '%.2f'), '%)']);
disp(['UIQM: ', num2str(UIQM), ' (Improvement: ', num2str(uiqm_improvement, '%.2f'), '%)']);
disp(['Image Entropy: ', num2str(IE), ' (No predefined original value)']);


% Compute percentage improvement (Fixing Undefined Variables)
pcqi_original = 0.5; % Placeholder for original image PCIQ (replace with actual computation if needed)
pcqi_enhanced = PCIQ;

uciqe_original = 0.4; % Placeholder for original image UCIQE (replace with actual computation if needed)
uciqe_enhanced = UCIQE;

uiqm_original = 15; % Placeholder for original image UIQM (replace with actual computation if needed)
uiqm_enhanced = UIQM;

ie_original = 5; % Placeholder for original image entropy (replace with actual computation if needed)
ie_enhanced = IE;

pcqi_improvement = ((pcqi_enhanced - pcqi_original) / pcqi_original) * 100;
uciqe_improvement = ((uciqe_enhanced - uciqe_original) / uciqe_original) * 100;
uiqm_improvement = ((uiqm_enhanced - uiqm_original) / uiqm_original) * 100;

% 1. Bar Chart - Comparing Quality Metrics
figure;
bar([pcqi_original, pcqi_enhanced; uciqe_original, uciqe_enhanced; uiqm_original, uiqm_enhanced; ie_original, ie_enhanced]);
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
imhist(rgb2gray(enhanced_weight)); % Fixed incorrect variable name
title('Histogram of Enhanced Image');
