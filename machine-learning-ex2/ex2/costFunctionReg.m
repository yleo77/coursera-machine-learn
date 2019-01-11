function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta = sigmoid(X * theta);

temp1 = -(y .* log(h_theta));
temp2 = -(1 - y) .* log(1 - h_theta);

thetaRegular = theta;
thetaRegular(1) = 0;

% 获取正则化逻辑回归项
rterm = lambda / (2 * m) * sum(thetaRegular .^ 2);
% 获取正则化参数
gterm = lambda / m * thetaRegular;

J = sum(temp1 + temp2) / m + rterm;

grad = (X' * (h_theta - y)) / m + gterm;

% =============================================================

end
