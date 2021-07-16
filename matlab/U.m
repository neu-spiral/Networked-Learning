function [out] = U(lambda, X, sigma, Sigma, T)
%Compute objective U by sampling
%   Detailed explanation goes here
[L,~] = size(lambda);
n_sample = 1000;
out = 0;

for sample = 1:n_sample
    N = poissrnd(lambda*T);
    for l = 1:L
        out = out + G(N(l,:), X, sigma, Sigma{l}) - G(0, X, sigma, Sigma{l});
    end
end
out = out/n_sample;

end