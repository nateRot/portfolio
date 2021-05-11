function [W, error] = adaline(X, Y, num_epochs, lr0)
% [W, error] = adaline(X, Y, num_epochs, lr0)
% Train a linear classifier using backpropogation
%
% Input:    
%   X, Y - training set (X is an data_dim x num_samples matrix, 
%                        Y is an 1 x num_samples vector)
%   lr - learning rate/step size (float)
%   num_epochs - max. number of epochs (int)
%
% Output:
%   w - weight vector
%   error - percent of training examples that were misclassified
%
% Last modified by Rotem Mulayoff 7/11/19

dataDim = size(X,1);
numSamples = size(X,2);

% Augment X to include a bias term
X = [X; ones(1,numSamples)];

% Initialize weights
W = 0.1*rand(dataDim+1, 1);

for n = 1:num_epochs
    for i = 1:numSamples
        % TODO: update W
        
        lr = lr0;
        Y_pred = W' * X(:,i);
        W = W - lr * X(:,i) * ( Y_pred - Y(i));
        
        % You may optionally uncomment the function below to see the change
        % followed by each example
        % plot_iris(X(1:2,:),Y, W)  
    end
    plot_iris(X(1:2,:),Y, W)  

end

% TODO: calculate the zero-one loss. 
% i.e. on what percent of the data do we make an error
Ypred = W' * X;
Ypred( Ypred < 0 ) = -1;
Ypred( Ypred > 0 ) = 1;
error = sum(Ypred ~= Y)/numSamples;

end

