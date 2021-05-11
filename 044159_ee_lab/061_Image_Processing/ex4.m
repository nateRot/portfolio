close all; clc;
cam_man = imread('cameraman.tif');
figure(1); imshow(cam_man);
cam_blur = imread('cam_blur.tif');
figure(2); imshow(cam_blur);
filter = fspecial('motion',7);
our_blur = imfilter(cam_man,filter,'replicate');
figure(3); imshow(our_blur);

deconv = deconvlucy(cam_blur,filter);
figure(4); imshow(deconv);

immse(cam_blur,our_blur)