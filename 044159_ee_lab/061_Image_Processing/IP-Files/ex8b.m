clc; close all;
im = imread('peppers.png');

figure; imshow(im); impixelinfo;

hsv=rgb2hsv(im); % Convert from RGB model to HSV model

h=hsv(:,:,1);
s=hsv(:,:,2);
v=hsv(:,:,3);

figure; imshow(h); impixelinfo;title('Hue');
figure; imshow(s); impixelinfo;title('Saturation');
figure; imshow(v); impixelinfo;title('Value');
 
mask=((h>=0.9) | (h<=0.14));
h(mask)=0.18;


hsv(:,:,1)=h;
 
newcolor = hsv2rgb(hsv);
figure; imshow(newcolor); impixelinfo;