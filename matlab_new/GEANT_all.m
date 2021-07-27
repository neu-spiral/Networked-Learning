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

% Remove the outcoming edges of the learners
for l = 1:L
    findl = find(TWINDi ~= Learner(l));
    TWINDi = TWINDi(findl);
    TWINDj = TWINDj(findl);
    TW = TW(findl);
end

% All features
X = abs(randn([p,d]));
% Source rates
lambda_s = 4*rand([S,p]);
sigma = rand;
%Sigma = diag(rand(d,1));
Sigma = cell(L,1);
for l = 1:L
    Sigma{l} = 2*diag(abs(randn(d,1)));
end

K = 100;
delta = 1/K;
n_sample = 40;
t_prime = 15;

% Find all incoming edges of learner No.1
Edge_1 = find(TWINDj == 1);

NC = 4;
Cap_Sum = cell(NC,1);
Cap_Alpha = cell(NC,1);
Cap_FW = cell(NC,1);
%Cap_FW_fast = cell(NC,1);
Cap_GP = cell(NC,1);

NS = 4;
S_Sum = cell(NS,1);
S_Alpha = cell(NS,1);
S_FW = cell(NS,1);
%Cap_FW_fast = cell(NS,1);
S_GP = cell(NS,1);

%% Increase source rate
for factor = 1:NS
% F-W

[S_Alpha{factor}, lambda_e1] = MaxAlpha(TWINDi,TWINDj,TW,V,lambda_s*factor,L,p,Learner,Source,5);

[S_Sum{factor}, lambda_e2] = MaxAlpha(TWINDi,TWINDj,TW,V,lambda_s*factor,L,p,Learner,Source,0);

tic
lambda_fw = zeros(L,p);
for k = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_fw, T, X, sigma, Sigma, t_prime, n_sample);    

    % Do LP, projection
    vk = LP(TWINDi,TWINDj,TW,V,lambda_s*factor,grad,Learner,Source);
    
    % Update solution
    lambda_fw = lambda_fw + delta * vk;
    k
    U(lambda_fw, X, sigma, Sigma, T)
    S_FW{factor} = lambda_fw;
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
lambda_pg = S_Alpha{factor};
for kk = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_pg, T, X, sigma, Sigma, t_prime, n_sample);    
    
    % Update solution
    lambda_pg = lambda_pg + delta * grad;
    
    % Projection
    lambda_pg = proj(TWINDi,TWINDj,TW,V,lambda_pg,grad,Learner,Source,lambda_s*factor);
    kk
    U(lambda_pg, X, sigma, Sigma, T)
    S_GP{factor} = lambda_pg;
end
time_PG = toc
end

%% Decrease learner link capacity
Cap_Alpha{1} = S_Alpha{1};
Cap_Sum{1} = S_Sum{1};
Cap_FW{1} = S_FW{1};
Cap_GP{1} = S_GP{1};

for cap = 2:NC
% F-W

TW(Edge_1) = TW(Edge_1)/2;

[Cap_Alpha{cap}, lambda_e1] = MaxAlpha(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,5);

[Cap_Sum{cap}, lambda_e2] = MaxAlpha(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,0);

tic
lambda_fw = zeros(L,p);
for k = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_fw, T, X, sigma, Sigma, t_prime, n_sample);    

    % Do LP, projection
    vk = LP(TWINDi,TWINDj,TW,V,lambda_s,grad,Learner,Source);
    
    % Update solution
    lambda_fw = lambda_fw + delta * vk;
    k
    U(lambda_fw, X, sigma, Sigma, T)
    Cap_FW{cap} = lambda_fw;
end
Cap_time_FW = toc

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
lambda_pg = Cap_Alpha{cap};
for kk = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_pg, T, X, sigma, Sigma, t_prime, n_sample);    
    
    % Update solution
    lambda_pg = lambda_pg + delta * grad;
    
    % Projection
    lambda_pg = proj(TWINDi,TWINDj,TW,V,lambda_pg,grad,Learner,Source,lambda_s);
    kk
    U(lambda_pg, X, sigma, Sigma, T)
    Cap_GP{cap} = lambda_pg;
end
Cap_time_PG = toc
end


save('GEANT_all_2.mat')