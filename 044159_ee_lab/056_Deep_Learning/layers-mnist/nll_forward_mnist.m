function loss = nll_forward_mnist(pi, y)
% loss = nll_forward_mnist(pi, y)
% This function calculates the negative log-likelihood loss of the examples. 
%   Inputs:
%       pi - probability distribution vectors of the samples 
%               (nClasses x nSamples), values between 0 and 1.
%       y - one-hot vectors of the true class of the samples
%               (nClasses x nSamples), values in {0,1}
%
%   Output:
%       loss - vector of loss for each example (1 x nSamples)
%
% Last modified by Rotem Mulayoff 7/11/19

if nargin < 2
    error("Number provide 3 inputs");
end

if size(pi,1) ~= size(y,1)
    error("dimensions of inputs mismatch")
end

if max(pi(:)) > 1 || min(pi(:)) < 0
    error("values of pi must be in range [0,1]")
end

if sum(sum(ismember(y,[0,1]))) ~= numel(y)
    error("values of target must belong to {0,1}")
end

% TODO: compute the loss function.
% NOTE: We want a loss value for each example. An external function will
% average the loss values.
 loss = -(sum(y.*log(pi)));

end

