function dE_dz_in = sigmoid_backward(z_in, dE_dz_out)
% dE_dz_in = sigmoid_backward(z_in, dE_dz_out)
% This function computes the gradient of the elementwise sigmoid function
%   Inputs:
%       z_in - the input of the sigmoid (nFeatures x nExamples)
%       dE_dz_out - gradient of output (nFeatures x nExamples)
%
%   Output:
%       dE_dz_in - gradient w.r.t. the inputs of the sigmoid (nFeatures x nExamples)
%
% Last modified by Rotem Mulayoff 7/11/19


% TODO: Compute the gradient of the sigmoid function w.r.t. it's inputs
% hint: you can use the forward function you already completed
dE_dz_in = dE_dz_out.*(sigmoid_forward(z_in)).*(1-sigmoid_forward(z_in));

end

