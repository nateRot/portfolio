close all; clc;
barbara = imread("barbara.tif");
figure(19);imshow(barbara);impixelinfo;
% N=16;
% levels=linspace(0,256,N+1);
% values=levels(1:end-1)+(128/N);
% im_qu=uint8(imquantize(barbara,levels(2:end-1),values));
% figure(3); imshow(im_qu);
% figure(4); imhist(im_qu);
% 
% N=16;
% [levels,values]=lloyds(double(barbara(:)),N);
% im_qo=uint8(imquantize(barbara,levels,values));
% figure(5); imshow(im_qo);
% figure(6); imhist(im_qo);
% 
% immse(barbara,im_qo);

figure(10);
b1 = imcrop(barbara,[179 6 15 15]);
subplot(2,2,1); imshow(b1); title('Block 1');
b2 = imcrop(barbara,[184 384 15 15]);
subplot(2,2,2); imshow(b2); title('Block 2');
b3 = imcrop(barbara,[420 466 15 15]);
subplot(2,2,3); imshow(b3); title('Block 3');
 
figure(11);
b1_dct=dct2(b1);
subplot(2,2,1); imshow(sqrt(abs(b1_dct)),[0 50]); title('DCT Block 1');
figure(11);
b2_dct=dct2(b2);
subplot(2,2,2); imshow(sqrt(abs(b2_dct)),[0 50]); title('DCT Block 2');
figure(11);
b3_dct=dct2(b3);
subplot(2,2,3); imshow(sqrt(abs(b3_dct)),[0 50]); title('DCT Block 3');

im_dct_b=blockproc(barbara,[8 8],@(block_struct)dct2(block_struct.data));
figure; imshow(abs(im_dct_b),[]); impixelinfo;

% sum(abs(im_dct_b(:))<10)/(512*512)

im_dct_b(abs(im_dct_b)<10)=0;
im_r=uint8(blockproc(im_dct_b,[8 8],@(block_struct)idct2(block_struct.data)));
figure(2); imshow(abs(im_r),[]); impixelinfo;
immse(im_r,barbara)