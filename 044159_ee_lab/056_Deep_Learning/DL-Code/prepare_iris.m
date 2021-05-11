function [Xtrain, Ytrain, Xtest, Ytest] = prepare_iris(meas, species, trainRatio, range)
%PREPARE_IRIS prepare the iris dataset for training: convert to numerical,
%             remove 'virginica', split to train and test
%   meas, species - iris measurements and label
%   range - the range of the labels: {"zero", "sign"}

% Remove all virginica entries from the dataset so there are only two
% classes (we want to solve a binary classification problem)
inds = ~strcmp(species, 'virginica');

% Use only the measurements of Sepal Width and Sepal Length
featureInds = 3:4;


X = meas(inds, featureInds)';
Y = species(inds);

% Convert Y from string to int
[Yint, ~] = grp2idx(Y);

if range == "sign"
    % shift Y to be {1,-1}
    Yint = 2*Yint - 3;
elseif range == "zero"
    Yint = Yint - 1;
else
    error("range must be 'zero' or 'sign'");
end
% convert to row matrix
Yint = Yint';

% split randomely data to train and test
dataSize = size(X, 2);
trainSize = ceil(trainRatio * dataSize);

randomIdxs = randperm(dataSize);

Xtrain = X(:,randomIdxs(:,1:trainSize));
Ytrain = Yint(randomIdxs(1:trainSize));
Xtest = X(:,randomIdxs(trainSize+1:end));
Ytest = Yint(randomIdxs(trainSize+1:end));
end

