function [X_norm, mu, sigma] = featureNormalize(X)

%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1.
% normalize the feature values to improve the convergence of gradient descent

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
n = length(X(1,:));
for i = 1 : n,
    mu(1,i) = mean(X_norm(:,i));
    sigma(1,i) = std(X_norm(:,i));
    X_norm(:,i) = (X_norm(:,i)-mu(1,i))/sigma(1,i);
end;

end
