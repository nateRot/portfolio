function [W, b, error] = logistic_regression(X, Y, num_epochs, lr0)
% [W, b, error] = logistic_regression(X, Y, num_epochs, lr0)
% Train a linear classifier using backpropogation
%
%   Inputs:
%       X, Y - training set (X is an data_dim x num_samples matrix, 
%                        Y is an 1 x num_samples vector)
%       num_epochs - max. number of epochs (int)
%       lr0 - initial learning rate/step size (float)
%
%   Outputs:
%       W - weights (vector)
%       b - bias (scalar)
%       error - percent of training examples that were misclassified
%
% Last modified by Rotem Mulayoff 7/11/19

dataDim = size(X,1);
numSamples = size(X,2);

% Initialize weights
W = 0.1*rand(dataDim, 1);
b = 0;

% Figure initialization
f = figure(2);
h = animatedline;
axis([1 num_epochs 0 1])
xlabel('epoch');
ylabel('loss');
title('Loss vs. epoch');
f.OuterPosition = [1,f.OuterPosition(2:end)];

for n = 1:num_epochs
    for ii = 1:numSamples
        lr = lr0/n;
               
        x = X(:,ii);
        y = Y(:,ii);
        
%       TODO: complete the forward pass for one sample
       wx = affine_forward(x ,W ,b);
       swx = sigmoid_forward(wx);

        
%       TODO: complete the backward pass
       dE_dswx = nll_backward(swx ,y);
       dE_dwx = sigmoid_backward(wx, dE_dswx);
       [~, dE_dw, dE_db] = affine_backward(x ,W ,b , dE_dwx);

        
        W = W - lr*dE_dw;
        b = b - lr*dE_db;
        
        % You may optionally uncomment the function below to see the change
        % followed by each example
        % plot_iris(X,Y,[W;b])
    end
    %  TODO: complete the average loss for the epoch on all the examples in the training dataset
      wx_full = affine_forward(X, W, b);
      swx_full = sigmoid_forward(wx_full);
      loss = nll_forward(swx_full, Y);
      avgloss = mean(loss);

    % Plot the loss value to the graph
    figure(2)
    addpoints(h,n, avgloss);
    drawnow
    plot_iris(X,Y,[W;b])

end

% TODO: calculate the zero-one loss. 
% i.e. on what percent of the data do we make an error
wx = affine_forward(X, W, b);
swx = sigmoid_forward(wx);
swx(swx>0.5) = 1;
swx(swx<0.5) = 0;
error = sum(Y~=swx)/length(Y);

end

