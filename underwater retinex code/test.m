%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Underwater Image Enhancement Based on Adaptive Color Correction and Improved Retinex Algorithm

close all; % Close all open figures
clear all; % Clear all variables from workspace
clc; % Clear the command window
warning off; % Suppress warnings

%% Image input
rgbImage = imread('27.jpg'); % Read the input image
figure, imshow(rgbImage), title('Original Image'); % Display the original image

rgbImage = im2double(rgbImage); % Convert image to double precision for mathematical processing
grayImage = rgb2gray(rgbImage); % Convert to grayscale for Retinex processing

%% Improved Retinex Algorithm
redChannel = rgbImage(:, :, 1); % Extract the red channel
greenChannel = rgbImage(:, :, 2); % Extract the green channel
blueChannel = rgbImage(:, :, 3); % Extract the blue channel

meanR = mean2(redChannel); % Compute mean intensity of red channel
meanG = mean2(greenChannel); % Compute mean intensity of green channel
meanB = mean2(blueChannel); % Compute mean intensity of blue channel
meanGray = mean2(grayImage); % Compute mean intensity of grayscale image

redChannel = (double(redChannel) * meanGray / meanR); % White balancing for red channel
greenChannel = (double(greenChannel) * meanGray / meanG); % White balancing for green channel
blueChannel = (double(blueChannel) * meanGray / meanB); % White balancing for blue channel

redChannel = redChannel - 0.3 * (meanG - meanR) .* greenChannel .* (1 - redChannel); % Suppress excess red
greenChannel = greenChannel; % Keep green channel unchanged
blueChannel = blueChannel + 0.3 * (meanG - meanB) .* greenChannel .* (1 - blueChannel); % Enhance blue channel

rgbImage_white_balance = cat(3, redChannel, greenChannel, blueChannel); % Merge corrected channels

figure('Name', 'Color Enhancement');
subplot(221); imshow(redChannel); title('Suppressed Red Channel'); % Display red channel after suppression
subplot(222); imshow(blueChannel); title('Enhanced Blue Channel'); % Display blue channel after enhancement
subplot(223); imshow(greenChannel); title('Green Channel'); % Display green channel
subplot(224); imshow(rgbImage_white_balance); title('After White Balance'); % Display final white-balanced image

%% Discrete Wavelet Transform (DWT) Processing
figure;
[LLr, LHr, HLr, HHr] = dwt2(rgbImage, 'haar'); % Perform 2D Haar wavelet decomposition
subplot(221), imshow(mat2gray(LLr)), title('LOW LOW Data'); % Show low-low subband
subplot(222), imshow(LHr), title('HIGH LOW Data'); % Show high-low subband
subplot(223), imshow(HLr), title('LOW HIGH Data'); % Show low-high subband
subplot(224), imshow(mat2gray(HHr)), title('HIGH HIGH Data'); % Show high-high subband

out = idwt2(LLr, LHr, HLr, HHr, 'haar'); % Perform inverse DWT
figure, imshow(mat2gray(out)); title('DWT Processed Image'); % Display reconstructed image

%% Retinex Processing with Gaussian Blur
img_filtered = rgb2gray(rgbImage_white_balance); % Convert white-balanced image to grayscale
img_blur = imgaussfilt(img_filtered, 15); % Apply Gaussian blur with kernel size 15
img_retinex = log(1 + img_filtered) - log(1 + img_blur); % Compute Retinex enhancement using log transform
img_retinex = imadjust(img_retinex, stretchlim(img_retinex), []); % Apply contrast stretching
figure, imshow(img_retinex); title('Retinex Enhanced Image'); % Display Retinex-processed image

%%% Optimal Weighted Fusion Retinex Process
under_image = rgbImage; % Initialize the working image
[luminance, saliency, chromatic] = size(rgbImage); % Get image dimensions

if chromatic == 3
    max_chromatic = rgb2ntsc(under_image); % Convert to NTSC color space for color correction
    max_chromatic(:, :, 2) = max_chromatic(:, :, 2) + (0.256 - max_chromatic(:, :, 2)); % Adjust chromatic channel 1
    max_chromatic(:, :, 3) = max_chromatic(:, :, 3) + (0.256 - max_chromatic(:, :, 3)); % Adjust chromatic channel 2
else
    max_chromatic = double(under_image) ./ 255; % Normalize grayscale image
end

under_image = max_chromatic .* 255; % Convert back to RGB scale
figure, imshow(under_image); title('Fusion Retinex Process Image'); % Display fused Retinex image

%% Temporal Correlation Pixels Separation (Contrast Enhancement)
Adaptive_red = adapthisteq(under_image(:, :, 1)); % Apply adaptive histogram equalization to red channel
adapthisteq_green = adapthisteq(under_image(:, :, 2)); % Apply adaptive histogram equalization to green channel
adapthisteq_blue = adapthisteq(under_image(:, :, 3)); % Apply adaptive histogram equalization to blue channel

enhanced_weight = cat(3, Adaptive_red, adapthisteq_green, adapthisteq_blue); % Merge enhanced channels
figure, imshow(mat2gray(enhanced_weight)); title('Proposed Enhanced Output Image'); % Display final enhanced image

%% Cropping and Sharpening
cropping = imcrop(enhanced_weight); % Allow user to select a region of interest
figure, imshow(cropping), title('Selected Part'); % Display selected region

h = fspecial('unsharp'); % Define unsharp masking filter
sharpended = imfilter(cropping, h, 'replicate'); % Apply unsharp filter to sharpen image
figure, imshow(sharpended); title('Sharpened Image'); % Display sharpened image
