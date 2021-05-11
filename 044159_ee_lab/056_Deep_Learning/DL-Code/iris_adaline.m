% Last modified by Rotem Mulayoff 7/11/19

clearvars; close all; clc;

%% Load the fisher iris dataset and set random seed
load fisheriris % if fails, install Statistics and Machine Learning Toolbox
rng(507);

%% Prepare data
trainRatio = 0.8;
[Xtrain, Ytrain, Xtest, Ytest] = prepare_iris(meas, species, trainRatio,...
                                              "sign");
X = [Xtrain,Xtest];
Y = [Ytrain,Ytest];

%% Set ADALINE's hyperparameters
numEpochs = 10;
lr0 = 0.02;

%% Train ADALINE to solve the task
[W, error] = adaline(Xtrain, Ytrain, numEpochs, lr0);
disp(['Train error rate = ', num2str(error), '%'])


%% Evaluate Test Error
% TODO: calculate the zero-one loss. 
% i.e. on what percent of the data do we make an error
testSize = size(Ytest,2);
Ypred = sign(W'*[Xtest;ones(1,testSize)]);
missmatch=Ytest-Ypred;
missmatch(missmatch~=0)=1;
error = sum(missmatch)/testSize;

disp(['Test error rate = ', num2str(error), '%'])
plot_iris(X, Y, W, Xtest);
