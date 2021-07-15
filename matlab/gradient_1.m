function grad = gradient_1(lambda, T, X, sigma, Sigma, t_prime, n_sample)
%GRADIENT Approximation of the gradient
[L,p] = size(lambda);

grad = zeros(L,p);

lambda_max = max(max(lambda))*T;

t_prime = ceil(lambda_max + max([t_prime, lambda_max]));

for l = 1:L
    for i = 1:p
        w_li = 0;
        for j = 1:n_sample
            n = zeros(p,1);
            for ii = 1:p
               n(ii) = poissrnd(lambda(l,ii)); 
            end
            for t = 1:t_prime
                nx = n;
                nx(i) = t+1;
                ny = n;
                ny(i) = t;
                w_li = w_li + (G(nx, X, sigma, Sigma{l}) - G(ny, X, sigma, Sigma{l})) * T * poisspdf(t,lambda(l,i)*T);
            end
        end
        %w_li
        grad(l,i) = w_li/n_sample;
    end
end
end

