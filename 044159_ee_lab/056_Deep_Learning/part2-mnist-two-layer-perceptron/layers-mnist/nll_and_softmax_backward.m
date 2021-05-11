function dE_dz_in = nll_and_softmax_backward(z_in, y)
% dE_dz_in = nll_and_softmax_backward(z_in, y)
% This function computes the derivation of softmax and cross-entropy functions.
%   Inputs:
%       z_in - input vector of the softmax (nClasses x nSamples)
%       y - one-hot vectors of the samples (nClasses x nSamples), values in {0,1}
%
%   Output:
%       dE_dz_in - gradient w.r.t. the input vector of the softmax (nClasses x nSamples)
%
% Last modified by Rotem Mulayoff 7/11/19

% TODO: compute the gradient of the nll and softmax
 dE_dz_in = softmax_forward(z_in)-y;


end

