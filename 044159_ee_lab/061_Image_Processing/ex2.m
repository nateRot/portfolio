lb=imread('liftingbody.png');
figure(1); imshow(lb);
figure(2); imhist(lb);
figure(3); histeq(lb);
figure(4); imhist(histeq(lb));