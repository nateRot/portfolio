% im = imread("rice.png");
% figure(1);imshow(im);impixelinfo;
% figure(3); imhist(im);
%  
% Thresh=125;
% bw=im>Thresh;
% figure(2);imshow(bw);impixelinfo;
% sum(im(:)>254)
% sum(bw(:))/(256*256)
% figure(4); imhist(bw);
% imfinfo('IP-Files/toucan.tif');

im = imread("pieces.png");
figure(1);imshow(im);impixelinfo;
Thresh=201;
bw=im<Thresh;
figure(2);imshow(bw);impixelinfo;
% 
% CC=bwconncomp(bw);
% labels=labelmatrix(CC);
% cmap=rand(1000,3);
% figure(3);imshow(labels,cmap);
% CC.NumObjects