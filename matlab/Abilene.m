% Abilene topology
TWINDi = [1 1  2 2  3 4 4  5 5 6 7 8 9  10   2 11 3 11 4 5 10 6 9 7 8 9 10 11 ];
TWINDj = [2 11 3 11 4 5 10 6 9 7 8 9 10 11   1 1  2 2  3 4 4  5 5 6 7 8 9  10 ];

% Link capacity
TW = 0.01*[856 2095 366 1295 1893 1176 902 846 587 233 700 260 548 639 856 2095 366 1295 1893 1176 902 846 587 233 700 260 548 639 ];
% DG = constructGraph(TWINDi,TWINDj,TW);
% h = view(biograph(DG));
% set(h,'ShowWeights','on');
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
lambda_s = 10*rand([S,p]);
sigma = rand;
%Sigma = diag(rand(d,1));
Sigma = cell(L,1);
for l = 1:L
    Sigma{l} = diag(rand(d,1));
end

K = 200;
delta = 1/K;
n_sample = 20;
t_prime = 10;

lambda_fw = 0.001*ones(L,p);
lambda_fwl = lambda_fw;

[v1, lambda_e1] = MaxAlpha(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,10);
sum(sum(v1))

[v2, lambda_e2] = MaxAlpha(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,0);
sum(sum(v2))

U(v2, X, sigma, Sigma, T)
U(v1, X, sigma, Sigma, T)

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
    U(lambda_fw, X, sigma, Sigma, T)
end
time_FW = toc

% % F-W with lesser samples
% tic
% for k = 1:K
%     % Estimate the current gradient
%     grad = gradient_1(lambda_fwl, T, X, sigma, Sigma, t_prime, 20);    
% 
%     % Do LP, projection
%     vk = LP(TWINDi,TWINDj,TW,V,lambda_s,grad,Learner,Source);
%     
%     % Update solution
%     lambda_fwl = lambda_fwl + delta * vk;
% end
% time_FWL = toc

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
    U(lambda_pg, X, sigma, Sigma, T)
end
time_PG = toc

save('Abilene.mat')