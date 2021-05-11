clc; close all; clear;
im = imread('plate08.jpg');
figure; imshow(im);impixelinfo;title('Original Image') 

%% Parameters
hue_threshold_max = 0.19;
hue_threshold_min = 0.045;
sat_threshold = 0.4;
Eccentricity_lvl = 0.9;

%% Clear as much as possible using HSV thresholds
hsv=rgb2hsv(im); % Convert from RGB model to HSV model

h=hsv(:,:,1);
s=hsv(:,:,2);
v=hsv(:,:,3);

subplot(2,2,1), imshow(h);impixelinfo; title('Hue')
subplot(2,2,2), imshow(s);impixelinfo; title('Saturation')
subplot(2,2,3), imshow(v);impixelinfo; title('Value')

% find threshold in HSV for plate
hue_mask=((h>=hue_threshold_min) & (h<=hue_threshold_max));
figure; imshow(hue_mask); impixelinfo;title('Hue Mask') 

sat_mask = (s >= sat_threshold);
bckg_mask_s = not(sat_mask);
figure; imshow(sat_mask); impixelinfo;title('Sat Mask') 

combined_mask = hue_mask.*sat_mask;
figure; imshow(combined_mask); impixelinfo;title('Sat & Hue Mask') 

% Remove leftover noise
SE1=strel('square',2);
eroded_mask =imerode(combined_mask,SE1);
figure; imshow(eroded_mask); impixelinfo;title('Eroded Mask') 

%% Begin extracting based on region properties
% Sort by eccentricity - how oval the shape is
stats=regionprops(eroded_mask>0,'all');

% find only indicies meeting Eccentricity requirment
Eccentric = [stats.Eccentricity];
[sorted_ecc, idx]=sort(Eccentric, 'descend');
Eccentric_idx = find(Eccentric > Eccentricity_lvl);

% Sort those meeting Eccentricity threshold by area
areas = [stats(Eccentric_idx).Area];
[sorted_area, sorted_idx] = sort(areas, 'descend');

% Indicies in stats that meet Eccentricity and sorted by area
Area_Ecc_idx = [Eccentric_idx(sorted_idx)];

% Get Euler numbers of the indices found and find obj with min Euler number
eulers = [stats(Area_Ecc_idx).EulerNumber];
[min_eulers, euler_idx] = min(eulers);
obj_idx = Area_Ecc_idx(euler_idx);
obj = stats(obj_idx);

%% Create mask fit for plate
% Use BoundingBox around object
right_x = round(obj.BoundingBox(1));
bottom_y = round(obj.BoundingBox(2));
left_x = ceil(right_x + obj.BoundingBox(3));
top_y = ceil(bottom_y + obj.BoundingBox(4));

% Create empty box
bounding_box = false(size(combined_mask));
% Select pixels to turn on
bounding_box([bottom_y:top_y], [right_x:left_x]) = true;

% Keep only pixels from bounding box
bound_mask = combined_mask.*bounding_box;
figure;imshow(bound_mask);title('Plate without noise') 

% Fill holes in mask - solid mask, lost by combined mask
filled_plate_mask = imfill(bound_mask, 'holes');
% figure;imshow(filled_plate_mask);title('Filled Mask') 

%% Complete extraction with filled licenses plate mask on RGB image
final_mask=uint8(repmat(filled_plate_mask,[1 1 3]));
result=im.*final_mask; 
figure; imshow(result); title('Final Extraction') 
