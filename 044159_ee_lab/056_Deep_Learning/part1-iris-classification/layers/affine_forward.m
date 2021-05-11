function Z = affine_forward(X, W, b)
% Z = affine_forward(X, W, b)
% This function performs affine transformation
%   Inputs:
%       X - input (nFeatures x nExamples)
%       W - weight matrix (nFeatures x newFeaturesSize)
%       b - bias vector (newFeaturesSize x nExamples)
%
%   Output:
%       Z - output (newFeaturesSize x nExamples)
%
% Last modified by Rotem Mulayoff 7/11/19

if nargin < 3
    error("Number provide 3 inputs");
end

if size(X,1) ~= size(W,1) || size(b,1) ~= size(W,2)
    error("dimensions of inputs mismatch")
end

% TODO: compute the affine layer function.
Z = W'*X + b;

end

