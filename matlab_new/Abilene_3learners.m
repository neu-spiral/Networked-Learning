% Abilene topology
TWINDi = [1 1  2 2  3 4 4  5 5 6 7 8 9  10   2 11 3 11 4 5 10 6 9 7 8 9 10 11 ];
TWINDj = [2 11 3 11 4 5 10 6 9 7 8 9 10 11   1 1  2 2  3 4 4  5 5 6 7 8 9  10 ];

% Link capacity
%TW = 0.1*[856 2095 366 1295 1893 1176 902 846 587 233 700 260 548 639 856 2095 366 1295 1893 1176 902 846 587 233 700 260 548 639 ];
TW = 50 + 100*rand(1,length(TWINDi));
% DG = constructGraph(TWINDi,TWINDj,TW);
% h = view(biograph(DG));
% set(h,'ShowWeights','on');
V = 11;

T=1;
p = 40;
d = 100;

L = 3;
S = 3;
Learner = [1 3 5];
Source = [2 6 8];

% Different types of learners
Tao = 2;
t_l = [1 2 2];

% Remove the outcoming edges of the learners
for l = 1:L
    findl = find(TWINDi ~= Learner(l));
    TWINDi = TWINDi(findl);
    TWINDj = TWINDj(findl);
    TW = TW(findl);
end

% All features
%X = 10*abs(randn([p,d]));
X = rand([p,d]);
for i = 1:p
    for j = 1:d
        if X(i,j)>0.7
            X(i,j) = 10 + 5*rand;
        end
    end
end

% Source rates, need to consider different types
lambda_s = zeros(S,p,Tao);
for tao = 1:Tao
    lambda_s(:,:,tao) = 5*rand([S,p]);
end

% We can also have different sigma for different type
sigma = 1;

% Initialize orthogonal prior covariance 
Help = zeros(L,d);
for l = 1:L
    Help(l,:) = abs(randn(1,d));
end
for i = 1:d
    temp = max(Help(:,i));
    for l = 1:L
        if Help(l,i) == temp
            Help(l,i) = 0.9 + 0.1*rand;
        else
            Help(l,i) = 0.1*rand;
        end
    end
end
Help = Help * 3;

Sigma = cell(L,1);
for l = 1:L
    Sigma{l} = diag(Help(l,:));
end

% Sigma = cell(L,1);
% for l = 1:L
%     Sigma{l} = 2*diag(abs(randn(d,1)));
% end

K = 100;
delta = 1/K;
n_sample = 30;
t_prime = 15;

% Find all incoming edges of learner No.1
Edge_1 = find(TWINDj == Learner(1));
Temp_edge_1 = TW(Edge_1);
% Find all incoming edges of learner No.2
Edge_2 = find(TWINDj == Learner(2));
Temp_edge_2 = TW(Edge_2);
% Find all incoming edges of learner No.3
Edge_3 = find(TWINDj == Learner(3));
Temp_edge_3 = TW(Edge_3);

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

[S_Alpha{factor}, lambda_e1] = MaxAlpha_2(TWINDi,TWINDj,TW,V,lambda_s*factor,L,p,Learner,Source,5,t_l);

[S_Sum{factor}, lambda_e2] = MaxAlpha_2(TWINDi,TWINDj,TW,V,lambda_s*factor,L,p,Learner,Source,0,t_l);

tic
lambda_fw = zeros(L,p);
for k = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_fw, T, X, sigma, Sigma, t_prime, n_sample);    

    % Do LP, projection
    vk = LP(TWINDi,TWINDj,TW,V,lambda_s*factor,grad,Learner,Source,t_l);
    
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
    lambda_pg = proj(TWINDi,TWINDj,TW,V,lambda_pg,grad,Learner,Source,lambda_s*factor,t_l);
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

TW(Edge_1) = Temp_edge_1/cap;
TW(Edge_2) = Temp_edge_2/cap;
TW(Edge_3) = Temp_edge_3/cap;

[Cap_Alpha{cap}, lambda_e1] = MaxAlpha_2(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,5,t_l);

[Cap_Sum{cap}, lambda_e2] = MaxAlpha_2(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,0,t_l);

tic
lambda_fw = zeros(L,p);
for k = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_fw, T, X, sigma, Sigma, t_prime, n_sample);    

    % Do LP, projection
    vk = LP(TWINDi,TWINDj,TW,V,lambda_s,grad,Learner,Source,t_l);
    
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
%lambda_pg = Initial(TWINDi,TWINDj,TW,V,L,p,lambda_s,Learner,Source,t_l);
lambda_pg = Cap_Alpha{cap};
for kk = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_pg, T, X, sigma, Sigma, t_prime, n_sample);    
    
    % Update solution
    lambda_pg = lambda_pg + delta * grad;
    
    % Projection
    lambda_pg = proj(TWINDi,TWINDj,TW,V,lambda_pg,grad,Learner,Source,lambda_s,t_l);
    kk
    U(lambda_pg, X, sigma, Sigma, T)
    Cap_GP{cap} = lambda_pg;
end
Cap_time_PG = toc
end


save('Abilene_learners2.mat')