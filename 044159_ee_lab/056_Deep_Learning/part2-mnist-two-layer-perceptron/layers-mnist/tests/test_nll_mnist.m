function test_nll_mnist
    addpath('..\');
    run(functiontests(localfunctions))
end

function testScalar(testCase)
    pi=[0.5; 0.5];
    target=[1; 0];

    expForward = -log(0.5);
    
    actForward = nll_forward_mnist(pi,target);
    
    verifyEqual(testCase,actForward,expForward);
end



function testScalarMultiInput(testCase)
    pi=[[0.5; 0.5] [0.5;0.5]];
    target=[[1; 0] [1; 0]];

    expForward = [-log(0.5) -log(0.5)];
    
    actForward = nll_forward_mnist(pi,target);
    
    verifyEqual(testCase,actForward,expForward);
end


% function testScalar2(testCase)
%     pi=[0.8; 0.2];
%     target=[1; 0];
% 
%     expForward = -log(0.8);
%     expBackward = (softmax_forward(pi)-target);
%     
%     actForward = nll_forward_mnist(pi,target);
%     actBackward = nll_and_softmax_backward(pi, target);
%         
%     verifyEqual(testCase,actForward,expForward);
%     verifyEqual(testCase,actBackward,expBackward, 'RelTol', 0.01);
% 
% end

function testScalar2MultiInput(testCase)
    pi=[[0.8; 0.2] [0.8; 0.2]];
    target=[[1; 0] [1; 0]];

    expForward = [-log(0.8) -log(0.8)];
    
    actForward = nll_forward_mnist(pi,target);
        
    verifyEqual(testCase,actForward,expForward);

end

function testRandomData(testCase)
    % Added by Rotem Mulayoff 25/08/19
    load('testDataB.mat','NLL');
    
    actForward = nll_forward_mnist(NLL.pi, NLL.target);
    actBackward = nll_and_softmax_backward(NLL.z_in, NLL.target);
    
    verifyEqual(testCase, actForward, NLL.loss,'RelTol',10e-10);
    verifyEqual(testCase, actBackward, NLL.dE_dz_in,'RelTol',10e-10);

end