im=imread('peppers.png'); 

hsv=rgb2hsv(im); % Convert from RGB model to HSV model
imnew=im; % Copy the image 'im' to another variable 'imnew'

hue=hsv(:,:,1);  sat=hsv(:,:,2); val=hsv(:,:,3);
% Hue channel, Saturation channel, Value channel

red=im(:,:,1);
green=im(:,:,2);
blue=im(:,:,3);

figure; title('Recognition of Colors');
subplot(3,4,1);imshow(im);title('Original RGB Image');
subplot(3,4,2);imshow(red);title('R (Red) of RGB Image');
subplot(3,4,3);imshow(green);title('G (Green) of RGB Image');
subplot(3,4,4);imshow(blue);title('B (Blue) of RGB Image');
subplot(3,4,5);imshow(hsv);title('Original in HSV coordinates');
subplot(3,4,6);imshow(hue);title('H (Hue) of Image');
subplot(3,4,7);imshow(sat);title('S (Saturation) of Image');
subplot(3,4,8);imshow(val);title('V (Value) of Image');
subplot(3,4,10);imshow((hue>0.95)|(hue<0.05));title('0<Hue<0.05 or 0.95<Hue<1');
subplot(3,4,11);imshow(sat>0.5);title('Saturation > 0.5');
subplot(3,4,12);imshow(val>0.5);title('Value > 0.5');