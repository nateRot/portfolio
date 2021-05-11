function dE_dpi = nll_backward(pi, y)
% dE_dpi = nll_backward(pi, y)
% This function calculates the gradient of the negative log-likelihood loss
% of the examples.
%   Inputs: 
%       pi - probabilities of the samples (1 x nSamples), values between 0 and 1.
%       y - the true class of the samples (1 x nSamples), values in {0,1}
%
%   Output:
%       dE_dpi - gradient w.r.t. the input vector of the NLL (1 x nSamples)
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

if sum(ismember(y,[0,1])) ~= numel(y)
    error("values of y must belong to {0,1}")
end


% TODO: compute the gradient of the NLL loss.
dE_dpi = (1-y)./(1-pi)-y./pi;

end
