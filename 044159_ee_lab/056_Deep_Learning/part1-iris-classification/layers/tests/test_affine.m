function test_affine
    addpath('..\');
    run(functiontests(localfunctions))
end

function testOnes(testCase)
    W=ones(3,1);
    b=1;
    X=ones(3,1);

    expForward = 4;
    expBackX = 4*ones(3,1);
    expBackW = 4*ones(3,1);
    expBackB = 4;
    
    actForward = affine_forward(X,W,b);
    [actBackX, actBackW, actBackB]= affine_backward(X,W,b,actForward);
    
    verifyEqual(testCase,actForward,expForward);
    verifyEqual(testCase,actBackX,expBackX);
    verifyEqual(testCase,actBackW,expBackW);
    verifyEqual(testCase,actBackB,expBackB);
end

function testOnesMultiInput(testCase)
    W=ones(3,1);
    b=1;
    X=[ones(3,1) ones(3,1)];

    expForward = [4 4];
    
    actForward = affine_forward(X,W,b);
    
    verifyEqual(testCase,actForward,expForward);
end

function testRange(testCase)
    W=(1:3)';
    b=7;
    X=(4:6)';

    expForward = 39;
    expBackX = expForward*(1:3)';
    expBackW = expForward*(4:6)';
    expBackB = expForward;
    
    actForward = affine_forward(X,W,b);
    [actBackX, actBackW, actBackB]= affine_backward(X,W,b,actForward);
    
    verifyEqual(testCase,actForward,expForward);
    verifyEqual(testCase,actBackX,expBackX);
    verifyEqual(testCase,actBackW,expBackW);
    verifyEqual(testCase,actBackB,expBackB);
end

function testRangeMultiInput(testCase)
    W=(1:3)';
    b=7;
    X=[(4:6)' (4:6)'];

    expForward = [39 39];

    actForward = affine_forward(X,W,b);
    
    verifyEqual(testCase,actForward,expForward);
end

function testAffineBackward(testCase)
    dE_dz = [7; 7];
    W=[(1:3); (1:3)]';
    b=7;
    X=(4:6)';

    expBackX = [14; 28; 42];
    expBackW = [28 28; 35 35; 42 42];
    expBackB = [7; 7];
        
    [actBackX, actBackW, actBackB]= affine_backward(X,W,b,dE_dz);
    
    verifyEqual(testCase,actBackX,expBackX);
    verifyEqual(testCase,actBackW,expBackW);
    verifyEqual(testCase,actBackB,expBackB);
end

function testRandomData(testCase)
    % Added by Rotem Mulayoff 25/08/19
    
    load('testDataA.mat','Affine');
    act_z = affine_forward(Affine.X, Affine.W, Affine.b);  
    [act_dE_dx, act_dE_dw, act_dE_db] = affine_backward(Affine.X, Affine.W,...
                                                Affine.b, Affine.dE_dz);
    
    
    verifyEqual(testCase, act_z, Affine.z,'RelTol',10e-10);
    verifyEqual(testCase, act_dE_dx, Affine.dE_dx,'RelTol',10e-10);
    verifyEqual(testCase, act_dE_dw, Affine.dE_dw,'RelTol',10e-10);
    verifyEqual(testCase, act_dE_db, Affine.dE_db,'RelTol',10e-10);
end

