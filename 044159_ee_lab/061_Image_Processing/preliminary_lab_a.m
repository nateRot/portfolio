
im = imread("IP-Files/peppers.png");

figure(1);imshow(im,[]); impixelinfo;

my_filter = fspecial('motion', 50, 45);

filteredRGB = imfilter(im, my_filter,'replicate');
figure(2), imshow(filteredRGB);

%BW = imbinarize(im);

%figure(2);imshow(BW); impixelinfo;

%figure(3);histogram(im);

%figure(4);histogram(BW);


immse(im,filteredRGB)