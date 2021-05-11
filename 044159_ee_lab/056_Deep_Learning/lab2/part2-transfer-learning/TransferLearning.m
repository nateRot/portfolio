%% TRANSFER LEARNING
% In this script, we will learn to classify desserts using transfer
% learning.

% Last modified by Rotem Mulayoff 7/11/19

clearvars; close all; clc;

%% Load food dataset

path = 'C:\Deep Learning experiment\TransferLearningDataset';
imds = imageDatastore(path, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Display data
num_datapoints = numel(imds.Files);

% Display some data points
figure;
perm = randperm(num_datapoints,20);
for ii = 1:20
    subplot(4,5,ii);
    imshow(imds.Files{perm(ii)});
end

% Display label count
labelCount = countEachLabel(imds)
numClasses = numel(unique(categories(imds.Labels)))

% Split data
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomize');

%% Load Pre-Trained Net
net = alexnet;
layers = net.Layers
inputSize = net.Layers(1).InputSize

%% Replace final layers

% TODO:
% seperate the first 22 layers (all but last 3)
% freeze the layers using freezeWeights function
layersTransfer = freezeWeights(layers(1:22));

layers = [
    layersTransfer
% 	dropoutLayer(0.1);
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];


%% Data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5]);

% Enable data augmentation here
imdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,'DataAugmentation',imageAugmenter);
    
%% Train network

opts = trainingOptions('adam', ...
    'MiniBatchSize',64, ...
    'MaxEpochs',12, ...
    'InitialLearnRate',0.5e-4, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',true, ...
    'Plots','training-progress');

% TODO: complete trainNetwork parameters
netTransfer = trainNetwork(imdsTrain,layers,opts);

%% Test network
[YPred,scores] = classify(netTransfer,imdsValidation);

% display some results
idx = randperm(numel(imdsValidation.Files),20);
figure;
for ii = 1:20
    subplot(4,5,ii)
    I = readimage(imdsValidation,idx(ii));
    if YPred(idx(ii)) == imdsValidation.Labels(idx(ii))
       colorText = 'g'; 
    else
       colorText = 'r';
    end
    imshow(I)
    label = YPred(idx(ii));
    title(strrep(string(label),'_',' '),'Color',colorText);
end

YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);
fprintf('Validation error: %2.2f%%\n', 100*(1-accuracy));

