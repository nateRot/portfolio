% Train the two-layer perceptron on the MNIST dataset and evaluate its
% performance.

% Last modified by Rotem Mulayoff 7/11/19

clear; clc; close all;

%% Prepare data

validation_percent = 0.1;
[imdsTrain,imdsValidation] = mnistDataPrep(validation_percent);

images = imdsTrain.readall;
labels = grp2idx(imdsTrain.Labels);

inputValues = zeros(28*28, size(images, 1));
targetValues = zeros(10, size(images, 1));
for ind = 1:size(images,1) 
    tmp = cell2mat(images(ind));
    inputValues(:,ind) = double(tmp(:))./255;
    targetValues(labels(ind), ind) = 1;
end

%% Train

% Choose form of MLP:
numberOfHiddenUnits = 700;

% Choose appropriate parameters.
learningRate = 0.1;

% Choose number of epochs
epochs = 10;

fprintf('Train two layer perceptron with %d hidden units.\n', numberOfHiddenUnits);
fprintf('Learning rate: %2.2f\n', learningRate);

[hiddenWeights, outputWeights, b1, b2, error] = trainTwoLayerPerceptron(numberOfHiddenUnits, inputValues, targetValues, epochs, learningRate);

fprintf('Final train error: %2.2f%%\n', error);

%% Validate

% Load validation set.
images = imdsValidation.readall;
labels = grp2idx(imdsValidation.Labels);

inputValues = zeros(28*28, size(images, 1));
targetValues = zeros(10, size(images, 1));
for ind = 1:size(images,1) 
    tmp = cell2mat(images(ind));
    inputValues(:,ind) = double(tmp(:))/255;
    targetValues(labels(ind), ind) = 1;
end

fprintf('Validation:\n');
validationError = validateTwoLayerPerceptron(hiddenWeights, outputWeights, b1, b2, inputValues, targetValues);

fprintf('Validation error: %2.2f%%\n', validationError);
