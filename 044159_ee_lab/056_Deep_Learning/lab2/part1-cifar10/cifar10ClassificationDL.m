% Last modified by Rotem Mulayoff 7/11/19
clearvars; close all; clc;

%% Set categories and sets

% Please note: these are 4 of the 10 categories available
% Feel free to choose which ever you like best!
categories = {'airplane','automobile','frog','cat','deer','dog','bird','horse','ship','truck'};

path = 'C:\Deep Learning experiment\CIFAR10\';
rootFolder = strcat(path, 'cifar10Train');
imdsTrain = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');

rootFolder = strcat(path, 'cifar10Test');
imdsTest = imageDatastore(fullfile(rootFolder, categories), ...
    'LabelSource', 'foldernames');

%% Display data
num_datapoints = numel(imdsTrain.Files);

% Display some data points
figure;
perm = randperm(num_datapoints,20);
for ii = 1:20
    subplot(4,5,ii);
    imshow(imdsTrain.Files{perm(ii)});
end

% Display data point parameters
img = readimage(imdsTrain,1);
disp("Each Image is in size:")
size(img)
disp("Each Image is of type:")
range(img(:))

% Display label count
labelCount = countEachLabel(imdsTrain)
numImageCategories = numel(categories);

%% Data augmentation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do not uncomment this part now. Do so only for question 1 Data augmentation part! %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 augmenter = imageDataAugmenter( ...
		'RandXReflection',true);%, ...
% 		'RandRotation', [-10 10]);
% 		'RandXScale',[1 2], ...
% 		
% 		'RandYScale',[1 2]);
%      'RandXReflection',true, ...
%      'RandXScale',[1 2]), ...
%     'RandYReflection',true, ...
%      'RandYScale',[1 2]

 imdsTrain = augmentedImageDatastore(size(readimage(imdsTrain,1)), imdsTrain, 'DataAugmentation', augmenter);

%% Model architecture

% Create the image input layer for 32x32x3 CIFAR-10 images
inputLayer = imageInputLayer(size(img));

% Convolutional layer parameters
filterSize = [5 5];
numFilters = 32;

middleLayers = [
    
    % The first convolutional layer has a bank of 32 5x5x3 filters. Add
    % symmetric padding of 2 pixels to ensure that image borders
    % are included in the processing. This is important to avoid
    % information at the borders being washed away too early in the
    % network.
    %% YOUR CODE HERE %%
	convolution2dLayer(5,numFilters,'Padding',[2 2 2 2])
    % Note that the third dimension of the filter can be omitted because it
    % is automatically deduced based on the connectivity of the network. In
    % this case because this layer follows the image layer, the third
    % dimension must be 3 to match the number of channels in the input
    % image.

    % Next add the ReLU layer:
    %% YOUR CODE HERE %%
	reluLayer;
    % Follow it with a max pooling layer that has a 3x3 spatial pooling area
    % and a stride of 2 pixels. This down-samples the data dimensions from
    % 32x32 to 15x15.
    %% YOUR CODE HERE %%
	maxPooling2dLayer(3,'Stride',2);
    % Repeat the 3 core layers to complete the middle of the network.
    %% YOUR CODE HERE %%
	convolution2dLayer(5,numFilters,'Padding',[2 2 2 2]);
    reluLayer;
	maxPooling2dLayer(3,'Stride',2);
	
    % Repeat the 3 core layers to complete the middle of the network.
    % instead of numFilters, use 2*numFilters
    %% YOUR CODE HERE %%
	convolution2dLayer(5,2*numFilters,'Padding',[2 2 2 2]);
    reluLayer;
	maxPooling2dLayer(3,'Stride',2);
% 	dropoutLayer(0.1);
];


finalLayers = [
    
    % Add a fully connected layer with 64 output neurons. The output size of
    % this layer will be an array with a length of 64.
    %% YOUR CODE HERE %%
	fullyConnectedLayer(64)
    % Add an ReLU non-linearity.
    %% YOUR CODE HERE %%
	reluLayer;
    % Add the last fully connected layer. At this point, the network must
    % produce the correct number of signals that can be used to measure whether the input image
    % belongs to one category or another. This measurement is made using the
    % subsequent loss layers.
    %% YOUR CODE HERE %%
% 	dropoutLayer(0.05);
	fullyConnectedLayer(10)
    % Add the softmax loss layer and classification layer. The final layers use
    % the output of the fully connected layer to compute the categorical
    % probability distribution over the image classes. During the training
    % process, all the network weights are tuned to minimize the loss over this
    % categorical distribution.
    %% YOUR CODE HERE %%
	softmaxLayer; 
    classificationLayer;
];

layers = [
    inputLayer
    middleLayers
    finalLayers
    ]


%% Training options

opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 2e-4,...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.05, ...
    'LearnRateDropPeriod', 12, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 100, ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',200, ...
    'Verbose', true, ...
    'VerboseFrequency', 25, ...
    'Plots','training-progress');


%% Train
[net, info] = trainNetwork(imdsTrain, layers, opts);
 
%% Toad test set

%test:
labels = classify(net, imdsTest);

figure;
perm = randi(length(imdsTest.Labels),20);
for ii = 1:20
    subplot(4,5,ii);
    im = imread(imdsTest.Files{perm(ii)});
    imshow(im);
    if labels(perm(ii)) == imdsTest.Labels(perm(ii))
       colorText = 'g'; 
    else
       colorText = 'r';
    end
    title(char(labels(perm(ii))),'Color',colorText);
end

accuracy = mean(labels == imdsTest.Labels);
fprintf('Validation error: %2.2f%%\n', 100*(1-accuracy));

