function [imdsTrain,imdsValidation] = mnistDataPrep(validation_percent)
%MNISTDATAPREP is a function that sets the digits data for classification
%with neural networks.

% Input:
% test_percent (0<double<1)- the percent of data that is we want as test
%                            dataset.
% Output:
% imdsTrain (imageDatastore)- train dataset
% imdsTest  (imageDatastore)- test dataset

% The dataset we will use is a dataset that is similar to MNIST, and
% contains images of written digits, each one labled with the written
% digit.

%% Load Dataset

% set dataset path
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
% read dataset into an imageDatastore
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% calculate number of datapoints
num_datapoints = numel(imds.Files);

%% Display data

% Display some data points
figure;
perm = randperm(num_datapoints,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

% Display data point parameters
img = readimage(imds,1);
disp("Each Image is in size:")
size(img)
disp("Each Image is of Type:")
range(img(:))

% Display label count
labelCount = countEachLabel(imds)

%% Split data- Train, Test

% set test percent out of total datapoints
% validation_percent = 0.1;
numTrainFiles = (1 - validation_percent)*mean(labelCount.Count);
% split dataset for each label
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
% labelCount = countEachLabel(imdsTrain)


end

