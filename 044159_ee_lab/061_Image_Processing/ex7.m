clc; close all;
% %  q a
% im=ones(3,1); im=padarray(im,[1 2]);
% SE1=strel('arbitrary',[1 1 1])
% out1=imdilate(im,SE1)
% SE2=strel('arbitrary',[1; 1; 1])
% out2=imdilate(im,SE2)
% % q b
% im=ones(4); im=padarray(im,[1 1]);
% SE1=strel('square',3)
% out1=imerode(im,SE1)
% SE2=strel('pair', [2 2])
% out2=imerode(im,SE2)

% % q c
% im=imread('shapes.jpg');
% figure(1); imshow(im,[]); title('original image');
% SE1=strel('disk',20);
% eroded=imerode(im,SE1);
% figure(2); imshow(eroded,[]); title('eroded image');
% final=imdilate(eroded,SE1);
% figure(3); imshow(final,[]); title('only circles');

% % q d
% im = imread("pieces.png");
% figure(1);imshow(im);impixelinfo;
% Thresh=201;
% bw=im<Thresh;
% figure(2);imshow(bw);impixelinfo;
% SE1 = strel('line',7,45)
% out=imerode(bw,SE1);
% SE2 = strel('line',7,45)
% out2=imdilate(out,SE2);
% figure(3); imshow(out2,[]); title('no screws');

% % q e
im = imread("rice.png");
figure(1);imshow(im);impixelinfo;
SE1=strel('disk',20);
out=imerode(im,SE1);
figure(2);imshow(out);impixelinfo;
im = im - out;
figure(3);imshow(im);impixelinfo;
figure(4); imhist(im);
Thresh=55;
bw=im>Thresh;
figure(45); imshow(bw);impixelinfo;
bw_r=bwpropfilt(bw,'Area',[150 300]);
figure; imshow(bw_r);
CC=bwconncomp(bw_r);
CC.NumObjects

sum(double(bw_r(:)))/CC.NumObjects

