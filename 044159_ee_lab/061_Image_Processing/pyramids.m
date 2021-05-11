clc; close all;
im=imread('pyramid.jpg');
im_crop = imread('pyramid.jpg');
im_crop(255:400, :, :) = 255;
im(1:254, :, :) = 0; 
hsv_im=rgb2hsv(im);
figure; imshow(im); impixelinfo;
figure; imshow(im_crop); impixelinfo;

hsv=rgb2hsv(im_crop); % Convert from RGB model to HSV model
x = 1:1:324;
size(hsv)
h=hsv(:,:,1);
size(h)
s=hsv(:,:,2);
v=hsv(:,:,3);

figure; imshow(h); impixelinfo;title('Hue');
figure; imshow(s); impixelinfo;title('Saturation');
figure; imshow(v); impixelinfo;title('Value');
 
mask =((h(:, 1:494) < 0.08));
h(mask)=0.70;


hsv(:,:,1)=h;

newcolor = hsv2rgb(hsv);
im = hsv2rgb(hsv_im);
newcolor(255:400, :, :) = 0;
figure; imshow(newcolor); impixelinfo;
im(1:254, :, :) = newcolor(1:254, :, :);
figure; imshow(im); impixelinfo;