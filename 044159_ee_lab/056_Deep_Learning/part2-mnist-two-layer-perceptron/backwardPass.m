function [dE_dw1, dE_db1, dE_dw2, dE_db2] = backwardPass(x, w1, b1, w2, b2, y, xw1, sxw1, sxw1w2)
% [dE_dw1, dE_db1, dE_dw2, dE_db2] = backwardPass(x, w1, b1, w2, b2, y, xw1, sxw1, sxw1w2)
% This function computes the gradients for all layers using backpropagation technique.
% Last modified by Rotem Mulayoff 7/11/19

% TODO: Complete the backward pass.
 dE_dsxw1w2 = nll_and_softmax_backward(sxw1w2,y);
 [dE_dsxw1, dE_dw2, dE_db2] = affine_backward(sxw1,w2,b2,dE_dsxw1w2);
 dE_dxw1 = sigmoid_backward(xw1,dE_dsxw1);
 [~, dE_dw1, dE_db1] = affine_backward(x,w1,b1,dE_dxw1);

end

