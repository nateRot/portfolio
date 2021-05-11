function [xw1,sxw1,sxw1w2,ssxw1w2,loss] = forwardPass(x, w1, b1, w2, b2, y)
% [xw1,sxw1,sxw1w2,ssxw1w2,loss] = forwardPass(x, w1, b1, w2, b2, y) 
% This function propagates the input vector through the network.
% Last modified by Rotem Mulayoff 7/11/19

% TODO: Complete the forward pass.
 xw1 = affine_forward(x,w1,b1);
 sxw1 = sigmoid_forward(xw1);
 sxw1w2 = affine_forward(sxw1,w2,b2);
 ssxw1w2 = softmax_forward(sxw1w2);
 loss = nll_forward_mnist(ssxw1w2,y);

end

