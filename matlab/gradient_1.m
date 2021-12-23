function grad = gradient_1(lambda, T, X, sigma, Sigma, t_prime, n_sample)
%GRADIENT Approximation of the gradient
[L,p] = size(lambda);

grad = zeros(L,p);

lambda_max = max(max(lambda))*T;

t_prime = ceil(lambda_max + max([t_prime, lambda_max]));

for l = 1:L
    for j = 1:n_sample
        n = poissrnd(lambda(l,:)*T);
        for i = 1:p
            w_li = 0;
            for t = 0:t_prime
                nx = n;
                nx(i) = t+1;
                ny = n;
                ny(i) = t;
                w_li = w_li + (G(nx, X, sigma, Sigma{l}) - G(ny, X, sigma, Sigma{l})) * T * poisspdf(t,lambda(l,i)*T);
            end
            if w_li < 0
                wli
            end
            grad(l,i) = grad(l,i) + w_li;
        end
    end
end

grad = grad/n_sample;

end

