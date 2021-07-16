% GEANT topology
TWINDi = [1 1 1  1  1  2 2 3 3 3 3  4 4 4  4  5 5  6 6 7 7 7 8 8 9 9  9  10 10 11 11 12 12 12 12 12 13 13 14 14 14 14 15 15 15 15 15 16 16 17 17 17 17 18 18 18 18 19 19 20 20 20 21 21 22 22];
TWINDj = [2 3 15 17 22 1 3 1 2 4 12 3 7 15 18 6 14 5 7 4 6 8 7 9 8 10 12 9  11 10 12 3  9  11 13 15 12 14 5  13 15 16 1  4  12 14 17 14 17 1  15 16 18 4  17 19 20 18 20 18 19 21 20 22 1  21];
TW = 15*rand(1,length(TWINDi));
%DG = constructGraph(TWINDi,TWINDj,TW);
%h = view(biograph(DG));
%set(h,'ShowWeights','on');
V=22;

T=1;
p = 20;
d = 100;

L = 2;
S = 3;
Learner = [1 5];
Source = [10 15 20];

% All features
X = rand([p,d]);
% Source rates
lambda_s = 10*rand([S,p]);
sigma = rand;
%Sigma = diag(rand(d,1));
Sigma = cell(L,1);
for l = 1:L
    Sigma{l} = diag(rand(d,1));
end

K = 100;
delta = 1/K;
n_sample = 20;
t_prime = 10;

lambda_fw = 0.001*ones(L,p);
lambda_fwl = lambda_fw;

[v1, lambda_e1] = MaxAlpha(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,10);
sum(sum(v1))

[v2, lambda_e2] = MaxAlpha(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,0);
sum(sum(v2))

% F-W
tic
for k = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_fw, T, X, sigma, Sigma, t_prime, n_sample);    

    % Do LP, projection
    vk = LP(TWINDi,TWINDj,TW,V,lambda_s,grad,Learner,Source);
    
    % Update solution
    lambda_fw = lambda_fw + delta * vk;
    k
end
time_FW = toc

% F-W with lesser samples
tic
for k = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_fwl, T, X, sigma, Sigma, t_prime, 10);    

    % Do LP, projection
    vk = LP(TWINDi,TWINDj,TW,V,lambda_s,grad,Learner,Source);
    
    % Update solution
    lambda_fwl = lambda_fwl + delta * vk;
end
time_FWL = toc

% Projected Gradient
tic
lambda_pg = v2;
for kk = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_pg, T, X, sigma, Sigma, t_prime, n_sample);    
    
    % Update solution
    lambda_pg = lambda_pg + delta * grad;
    
    % Projection
    lambda_pg = proj(TWINDi,TWINDj,TW,V,lambda_pg,grad,Learner,Source,lambda_s);
    kk
end
time_PG = toc

save('GEANT.mat')