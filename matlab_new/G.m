function [out] = G(n, X, sigma, Sigma)
%Function G
%   Detailed explanation goes here

out = log(det(X' * diag(n) * X + sigma^2 * Sigma^-1));

end

