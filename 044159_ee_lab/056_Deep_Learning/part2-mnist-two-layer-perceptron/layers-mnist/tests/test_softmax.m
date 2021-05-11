function test_softmax
    addpath('..\');
    run(functiontests(localfunctions))
end

function testOnes(testCase)
    X=zeros(3,1);

    expForward = 0.3333*ones(3,1);
    
    actForward = softmax_forward(X);
    
    verifyEqual(testCase,actForward,expForward, 'RelTol', 0.01);
end

function testOnesMultiInput(testCase)
    X=[zeros(3,1) zeros(3,1)];

    expForward = [0.3333*ones(3,1) 0.3333*ones(3,1)];
    
    actForward = softmax_forward(X);
    
    verifyEqual(testCase,actForward,expForward, 'RelTol', 0.01);
end

function testRandomData(testCase)
    % Added by Rotem Mulayoff 25/08/19
    
    load('testDataB.mat','Softmax');
    act_z_out = softmax_forward(Softmax.z_in);
    
    verifyEqual(testCase, act_z_out, Softmax.z_out,'RelTol',10e-10);
end