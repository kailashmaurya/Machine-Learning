function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%   GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    temp = (theta'*X')'.-y;
    inter1 = sum(temp);
    temp2 = temp.*X(:,2);
    inter2 = sum(temp2);
    temp3 = temp.*X(:,3);
    inter3 = sum(temp3);
    inter1 = (inter1*alpha)/m;
    inter2 = (inter2*alpha)/m;
    inter3 = (inter3*alpha)/m;
    theta(1) = theta(1) - inter1;
    theta(2) = theta(2) - inter2;
    theta(3) = theta(3) - inter2;

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
