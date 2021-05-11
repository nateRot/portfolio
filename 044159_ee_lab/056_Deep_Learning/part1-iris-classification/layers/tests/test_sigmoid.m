function test_sigmoid
    addpath('..\');
    run(functiontests(localfunctions))
end

function testOnes(testCase)
    X=zeros(3,1);

    expForward = 0.5*ones(3,1);
    expBackX = 0.25*ones(3,1).*expForward;
    
    actForward = sigmoid_forward(X);
    [actBackX]= sigmoid_backward(X,actForward);
    
    verifyEqual(testCase,actForward,expForward);
    verifyEqual(testCase,actBackX,expBackX);
end

function testOnesMultiInput(testCase)
    X=[zeros(3,1) zeros(3,1)];

    expForward = [0.5*ones(3,1) 0.5*ones(3,1)];
    
    actForward = sigmoid_forward(X);
    
    verifyEqual(testCase,actForward,expForward);
end

function testRandomData(testCase)
    % Added by Rotem Mulayoff 25/08/19
    
    load('testDataA.mat','Sigmoid');
    act_z_out = sigmoid_forward(Sigmoid.z_in);
    act_dE_dz_in = sigmoid_backward(Sigmoid.z_in, Sigmoid.dE_dz_out);
    
    verifyEqual(testCase, act_z_out, Sigmoid.z_out,'RelTol',10e-10);
    verifyEqual(testCase, act_dE_dz_in, Sigmoid.dE_dz_in,'RelTol',10e-10);
end