function z_out = sigmoid_forward(z_in)
% z_out = sigmoid_forward(z)
% This function performs elementwise sigmoid function on the input z
%   Input:
%       z_in - input of the sigmoid (nFeatures x nExamples)
%
%   Output:
%       z_out - output of the sigmoid (nFeatures x nExamples)
%
% Last modified by Rotem Mulayoff 7/11/19


% TODO: compute the sigmoid function.
z_out = 1./(1+exp(-z_in));

end

