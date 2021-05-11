function [dE_dx, dE_dW, dE_db] = affine_backward(X, W, b, dE_dz)
% [dE_dx, dE_dw, dE_db] = affine_backward(X, W, b, dE_dz)
% This function computes the gradients of an affine transformation.
%   Inputs:
%       X - input (nFeatures x nExamples)
%       W - weight matrix (nFeatures x newFeaturesSize)
%       b - bias vector (newFeaturesSize x nExamples)
%       dE_dz - gradient of output (newFeaturesSize x nExamples)
% 
%   Outputs:
%       dE_dx - gradient w.r.t. x (nFeatures x nExamples)
%       dE_dW - gradient w.r.t. weight matrix (nFeatures x newFeaturesSize)
%       dE_db - gradient w.r.t. bias vector (newFeaturesSize x nExamples)
%
% Last modified by Rotem Mulayoff 7/11/19

% TODO: Compute the gradients of an affine layer.
dE_dx = W*dE_dz;
dE_dW = X*dE_dz';
dE_db = dE_dz;

end