% Last modified by Rotem Mulayoff 7/11/19

clearvars; close all; clc;
addpath('./layers')

%% Load the fisher iris dataset and set random seed
load fisheriris
rng(507);

%% Prepare data
trainRatio = 0.8;
[Xtrain, Ytrain, Xtest, Ytest] = prepare_iris(meas, species, trainRatio,...
                                              "zero");
X = [Xtrain,Xtest];
Y = [Ytrain,Ytest];

%% Train network to solve the task
numEpochs = 20;
lr0 = 0.1;
[W, b, error] = logistic_regression(Xtrain, Ytrain, numEpochs, lr0);
fprintf('Train error = %2.2f%%\n', error);

%% Evaluate Test Error
% TODO: calculate the zero-one loss. 
% i.e. on what percent of the data do we make an error
testSize = size(Ytest,2);
wx = affine_forward(Xtest, W, b);
swx = sigmoid_forward(wx);
swx(swx>0.5) = 1;
swx(swx<0.5) = 0;
error = sum(Ytest~=swx)/length(Ytest);

fprintf('Test error = %2.2f%%\n', error);

plot_iris(X, Y, [W;b], Xtest);

