close all; clc;
church = imread("church.tif");
figure(1);imshow(church);impixelinfo;
% imedge=edge(church,'SOBEL',0,'both');
% figure(2); imshow(imedge,[]); title('SOBEL, Thresh=0');
% imedge=edge(church,'SOBEL',0.09,'both');
% figure(3); imshow(imedge,[]); title('SOBEL, Thresh=0.09');
% imedge=edge(church,'SOBEL',0.2,'both');
% figure(4); imshow(imedge,[]); title('SOBEL, Thresh=0.2');

% imedge=edge(church,'CANNY',0.35,0.1);
% figure(5); imshow(imedge,[]); title('CANNY, Thresh=0.35, Sigma=0.1');
% imedge=edge(church,'CANNY',0.35,1);
% figure(6); imshow(imedge,[]); title('CANNY, Thresh=0.35, Sigma=1');
% imedge=edge(church,'CANNY',0.35,5);
% figure(7); imshow(imedge,[]); title('CANNY, Thresh=0.35, Sigma=5');

% imedge=edge(church,'SOBEL',0.1,'both');
% figure(8); imshow(imedge,[]); title('SOBEL');
% imedge=edge(church,'LOG',0.017,1.5);
% figure(9); imshow(imedge,[]); title('LoG');
imedge=edge(church,'CANNY',0.25,1);
figure(10); imshow(imedge,[]); title('CANNY');

noise=std_n*randn(size(church));
im_n=uint8(double(church)+noise);
figure(12); imshow(im_n,[]); title('CANNY');
imedge=edge(im_n,'CANNY',0.25,1);
figure(11); imshow(imedge,[]); title('CANNY');