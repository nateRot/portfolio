function plot_iris(X, Y, W, Xtest)
%PLOT_IRIS Plots the iris dataset and seperator
%   X - all the datapoint (data_dim x num_samples)
%   Y - labels (data_dim x num_samples matrix)
%   Xtest - the points that belong to the test set
%   W - the weights of the linear seperator (data_dim)

figure(1);
gscatter(X(1,:),X(2,:),Y);
hold on
x = linspace(0,5.5);
% Convert W to "regular form"
% from W_1*x_1 + W_2*x_2 + W_3*1 = 0
% to x_2 = (-1/W_2)*(W_1*x_1 + W_3*1)
separator = -(W(1)*x+W(3))/W(2);
plot(x, separator)
if exist('Xtest', 'var')
    plot(Xtest(1,:),Xtest(2,:),'ko','MarkerSize',10)
end
xlim([0,5.5]); ylim([-0.5,2]);
xlabel('Sepal Length (cm)')
ylabel('Sepal Width (cm)')
if ~exist('Xtest', 'var')
legend('versicolor','setosa','ADALINE separator',...
       'Location', 'southeast')
else
    legend('versicolor','setosa','ADALINE separator',...
       'test set', 'Location', 'southeast')
end
hold off
end

