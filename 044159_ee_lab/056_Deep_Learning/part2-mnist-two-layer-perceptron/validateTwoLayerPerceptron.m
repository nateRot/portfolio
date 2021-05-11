function error = validateTwoLayerPerceptron(w1, w2, b1, b2, X, Y)
% validateTwoLayerPerceptron Validate the two layers perceptron using the
% validation set.
%
% INPUT:
% w1                             : Weights of the hidden layer.
% w2                             : Weights of the output layer.
% X                              : Input values for training (784 x 1000).
% Y                              : Labels for validation (10 x 1000).
%
% OUTPUT:
% error                          : 0-1 error
%
% Last modified by Rotem Mulayoff 7/11/19

addpath('../part1-iris-classification/layers')

setSize = size(X,2);
[~,~,~,ssxw1w2,~] = forwardPass(X, w1, b1, w2, b2, Y);
Y_hat = zeros(size(ssxw1w2));
[~,max_inds] = max(ssxw1w2);
linearInd = sub2ind(size(ssxw1w2), max_inds, 1:setSize);
Y_hat(linearInd) = 1;
error = 100*mean(sum(abs(Y_hat-Y))/2);

% Display some data points
figure;
perm = randperm(setSize,20);
for ii = 1:20
    subplot(4,5,ii);
    imshow(255*reshape(X(:,perm(ii)),28,28),[]);
    y = find(Y_hat(:,perm(ii)));
    y_gt = find(Y(:,perm(ii)));
    if y == y_gt
       colorText = 'g'; 
    else
       colorText = 'r';
    end
    title(num2str(y-1),'Color',colorText);
end

end