function [beta_MAP] = Beta(lambda, X, sigma, Sigma, T, beta, exp, l)
%Reture MAP Beta
%   Detailed explanation goes here
if exp == 1
    N = ceil(lambda(l,:)*T);
else
    N = poissrnd(lambda(l,:)*T);
end

[p,d] = size(X);

%beta_MAP = cell(1,L);


% for l = 1:L
%     % MAP estimator
%     X_M = zeros(sum(N(l,:)),d);
%     count = 0;
%     for i = 1:p
%         for j = 1:N(l,i)
%             count = count + 1;
%             X_M(count,:) = X(i,:);
%         end
%     end 
%     beta_MAP{l} = (X_M' * X_M + sigma^2 * Sigma{l}^-1)^-1 * (X_M') * (X_M * beta + normrnd(0,sigma,[sum(N(l,:)),1])); 
% end
        
% MAP estimator
X_M = zeros(sum(N),d);
count = 0;
for i = 1:p
    for j = 1:N(i)
        count = count + 1;
        X_M(count,:) = X(i,:);
    end
end 
beta_MAP = (X_M' * X_M + sigma^2 * Sigma{l}^-1)^-1 * (X_M') * (X_M * beta + normrnd(0,sigma,[sum(N),1])); 

end

