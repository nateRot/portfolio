function pi = softmax_forward(z_in)
% pi = softmax_forward(z_in)
% This function computes the softmax for multiclass classification task
%   Input:
%       z_in - input to the softmax layer (nClasses x nSamples)
%
%   Output:
%       pi - probability distribution vectors of the samples 
%               (nClasses x nSamples), values between 0 and 1.
%
% Last modified by Rotem Mulayoff 7/11/19


% TODO: complete the softmax layer function.
 pi = exp(z_in)./(sum(exp(z_in)));

end

