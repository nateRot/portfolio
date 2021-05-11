function test_nll
    addpath('..\');
    run(functiontests(localfunctions))
end

function testScalar(testCase)
    pi=0.5;
    target=1;

    expForward = -log(0.5);
    expBackPi = -2;
    
    actForward = nll_forward(pi,target);
    [actBackPi]= nll_backward(pi,target);
    
    verifyEqual(testCase,actForward,expForward);
    verifyEqual(testCase,actBackPi,expBackPi);
end

function testScalarMultiInput(testCase)
    pi=[0.5 0.5];
    target=[1 1];

    expForward = [-log(0.5) -log(0.5)];
    
    actForward = nll_forward(pi,target);
    
    verifyEqual(testCase,actForward,expForward);
end


function testScalar2(testCase)
    pi=0.5;
    target=0;

    expForward = -log(0.5);
    expBackPi = 2;
    
    actForward = nll_forward(pi,target);
    actBackPi = nll_backward(pi,target);
    
    verifyEqual(testCase,actForward,expForward);
    verifyEqual(testCase,actBackPi,expBackPi, 'RelTol', 0.01);
end

function testScalar2MultiInput(testCase)
    pi=[0.5 0.5];
    target=[0 0];

    expForward = [-log(0.5) -log(0.5)];
    
    actForward = nll_forward(pi,target);
    
    verifyEqual(testCase,actForward,expForward);
end

function testRandomData(testCase)
    % Added by Rotem Mulayoff 25/08/19
    
    load('testDataA.mat','NLL');
    actForward = nll_forward(NLL.pi, NLL.target);
    actBackPi = nll_backward(NLL.pi, NLL.target);
    
    verifyEqual(testCase, actForward, NLL.loss,'RelTol',10e-10);
    verifyEqual(testCase, actBackPi, NLL.dE_dpi,'RelTol',10e-10);
end
