
im=imread('IP-Files/toucan.tif');
figure(1); imshow(im);impixelinfo;
imfinfo('IP-Files/toucan.tif');

minvalue=min(min(im));
maxvalue=max(max(im));

meanvalue = mean(mean(im));

sumvalue = sum(sum(im==18));
 
L=im(170,:);
x = linspace(0,255,256);
figure(2);plot(x,L)
 
 
figure(2);imhist(im);
 
std_n=15;
noise=std_n*randn(size(im));
im_n=uint8(double(im)+noise);
figure(3); imshow(im_n);
figure(4); imhist(im_n);

wh=imread('IP-Files/whitehouse.tif');
figure(5); imshow(wh);impixelinfo;

wh(wh>230)=0;
figure(6); imshow(wh);impixelinfo;