function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    temp = (theta'*X')'.-y;
    inter1 = sum(temp);
    temp = temp.*X(:,2);
    inter2 = sum(temp);
    inter1 = (inter1*alpha)/m;
    inter2 = (inter2*alpha)/m;
    theta(1) = theta(1) - inter1;
    theta(2) = theta(2) - inter2;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %fprintf('Iteration = %d\nJ(theta) = %f\n\n', iter, J_history(iter));

end

end
