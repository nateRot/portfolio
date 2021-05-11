clc; close all;
bw = imread('objects.bmp');
figure; imshow(bw);

stats=regionprops(bw,'all');
stats(5)
stats(6)


f1=[stats.MajorAxisLength];
f2=[stats.ConvexArea];
figure(3); plot(f1,f2,'x');
title('Features 1');
xlabel('MajorAxisLength');
ylabel('ConvexArea');

f1=[stats.Perimeter];
f2=[stats.Solidity];
figure(4); plot(f1,f2,'x');
title('Features 2');
xlabel('Perimeter');
ylabel('Solidity');

f1=[stats.MinorAxisLength];
f2=[stats.Eccentricity];
figure(5); plot(f1,f2,'x');
title('Features 3');
xlabel('MinorAxisLength');
ylabel('Eccentricity');