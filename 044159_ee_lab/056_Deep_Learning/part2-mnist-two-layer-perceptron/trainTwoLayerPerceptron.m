function [w1, w2, b1, b2, error] = trainTwoLayerPerceptron(numberOfHiddenUnits, X, Y, epochs, learningRate)
% [w1, w2, b1, b2, error] = trainTwoLayerPerceptron(numberOfHiddenUnits, X, Y, epochs, learningRate)
% This function creates a two-layer perceptron and trains it on the MNIST dataset.
%
% INPUT:
% numberOfHiddenUnits            : Number of hidden units.
% X                              : Input values for training (28*28 x 9000)
% Y                              : Target values for training (10 x 9000)
% epochs                         : Number of epochs to train.
% learningRate                   : Learning rate to apply.
%
% OUTPUT:
% hiddenWeights                  : Weights of the hidden layer.
% outputWeights                  : Weights of the output layer.
%
% Last modified by Rotem Mulayoff 7/11/19

addpath('../part1-iris-classification/layers')
addpath('./layers-mnist')

% The number of training vectors.
trainingSetSize = size(X, 2);

% Input vector has 784 dimensions.
inputDimensions = size(X, 1);
% We have to distinguish between 10 digits.
outputDimensions = size(Y, 1);

% Initialize the weights for the hidden layer and the output layer.
w1 = normrnd(0, 2/(inputDimensions + numberOfHiddenUnits), inputDimensions, numberOfHiddenUnits);
w2 = normrnd(0, 2/(numberOfHiddenUnits + outputDimensions), numberOfHiddenUnits, outputDimensions);
b1 = zeros(numberOfHiddenUnits, 1);
b2 = zeros(outputDimensions, 1);

f = figure;
h = animatedline;
xlim([1,epochs]); ylim([0,1]);
xlabel('epoch');
ylabel('loss');
title('Loss vs. epoch')
f.OuterPosition = [1,f.OuterPosition(2:end)];

for t = 1: epochs
    disp(['epoch #', num2str(t)])
    for n = randperm(trainingSetSize)

        x = X(:,n);
        y = Y(:,n);

        % TODO: complete a training pass of the network 
         [xw1, sxw1, sxw1w2, ssxw1w2, ~] = forwardPass(x, w1, b1, w2, b2, y);
         [dE_dw1, dE_db1, dE_dw2, dE_db2] = backwardPass(x, w1, b1, w2, b2, y, xw1, sxw1, sxw1w2);
        

        w2 = w2 - learningRate.*dE_dw2;
        b2 = b2 - learningRate.*dE_db2;
        w1 = w1 - learningRate.*dE_dw1;
        b1 = b1 - learningRate.*dE_db1;

    end        

    % TODO: calculate the the loss avarage on all examples in the dataset
     [~,~,~,ssxw1w2,loss] = forwardPass(X, w1, b1, w2, b2, Y);
      avgloss = mean(loss);
    
    % plot the loss function
    addpoints(h,t, avgloss);
    drawnow

    % TODO: calculate the classification error
    % i.e. on what percent of the data do we make an error
     Y_hat = zeros(size(ssxw1w2));
     [~,max_inds] = max(ssxw1w2);
     linearInd = sub2ind(size(ssxw1w2), max_inds, 1:trainingSetSize);
     Y_hat(linearInd) = 1;
     error = 100*(sum(sum(Y_hat~=Y)/2)/trainingSetSize);

    % Print the training classification error to the screen
    fprintf('Train error = %2.2f%%\n', error);
end
    
end