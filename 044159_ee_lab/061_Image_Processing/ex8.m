clc; close all;
im = imread('peppers.png');
figure;imshow(im);
lab=rgb2lab(im);
L=lab(:,:,1);
a=lab(:,:,2);
b=lab(:,:,3);

% figure; imshow(L,[]);
% figure; histogram(L(:),300);
% figure; imshow(a,[]);
% figure; histogram(a(:),300);
% figure; imshow(b,[]);
% figure; histogram(b(:),300);

a_new=-128*ones(size(a));
lab_new=lab;
lab_new(:,:,2)=a_new;
rgb_new=lab2rgb(lab_new);
figure; imshow(rgb_new,[]);