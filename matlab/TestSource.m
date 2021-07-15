% We increase the rate of all sources

% Abilene topology
TWINDi = [1 1  2 2  3 4 4  5 5 6 7 8 9  10   2 11 3 11 4 5 10 6 9 7 8 9 10 11 ];
TWINDj = [2 11 3 11 4 5 10 6 9 7 8 9 10 11   1 1  2 2  3 4 4  5 5 6 7 8 9  10 ];

% Link capacity
TW = 0.01*[856 2095 366 1295 1893 1176 902 846 587 233 700 260 548 639 856 2095 366 1295 1893 1176 902 846 587 233 700 260 548 639 ];
%DG = constructGraph(TWINDi,TWINDj,TW);
V = 11;

T=1;
p = 20;
d = 100;

L = 2;
S = 3;
Learner = [1 5];
Source = [2 4 8];

% All features
X = rand([p,d]);
% Source rates
lambda_s = rand([S,p]);
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

C = 4;
Lambda_Sum = cell(C,1);
Lambda_Alpha = cell(C,1);
Lambda_FW = cell(C,1);
Lambda_FW_fast = cell(C,1);
Lambda_GP = cell(C,1);

tic
for factor = 1:C

lambda_s = lambda_s * 2;
    
lambda_fw = 0.001*ones(L,p);
lambda_fwl = lambda_fw;

[Lambda_Alpha{factor,1}, lambda_e1] = MaxAlpha(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,10);

[Lambda_Sum{factor,1}, lambda_e2] = MaxAlpha(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,0);

% F-W
for k = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_fw, T, X, sigma, Sigma, t_prime, n_sample);    

    % Do LP, projection
    vk = LP(TWINDi,TWINDj,TW,V,lambda_s,grad,Learner,Source);
    
    % Update solution
    lambda_fw = lambda_fw + delta * vk;
    
end

Lambda_FW{factor,1} = lambda_fw;

% F-W with lesser samples
for k = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_fwl, T, X, sigma, Sigma, t_prime, 10);    

    % Do LP, projection
    vk = LP(TWINDi,TWINDj,TW,V,lambda_s,grad,Learner,Source);
    
    % Update solution
    lambda_fwl = lambda_fwl + delta * vk;
end

Lambda_FW_fast{factor,1} = lambda_fwl;

% Projected Gradient
lambda_pg = Lambda_Sum{factor,1};
for kk = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_pg, T, X, sigma, Sigma, t_prime, n_sample);    
    
    % Update solution
    lambda_pg = lambda_pg + delta * grad;
    
    % Projection
    lambda_pg = proj(TWINDi,TWINDj,TW,V,lambda_pg,grad,Learner,Source,lambda_s);
    
end

Lambda_GP{factor,1} = lambda_pg;
toc
end
save('Abilene_source.mat')