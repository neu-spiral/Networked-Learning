function [Cap_Sum,Cap_Alpha,Cap_FW,Cap_GP,S_Sum,S_Alpha,S_FW,S_GP] = main_simulation(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,t_l, T, X, sigma, Sigma, t_prime, n_sample,K,delta)

% Find all incoming edges of learner No.1
Edge_1 = find(TWINDj == Learner(1));
Temp_edge_1 = TW(Edge_1);
% Find all incoming edges of learner No.2
Edge_2 = find(TWINDj == Learner(2));
Temp_edge_2 = TW(Edge_2);

NC = 4;
Cap_Sum = cell(NC,1);
Cap_Alpha = cell(NC,1);
Cap_FW = cell(NC,1);
Cap_GP = cell(NC,1);

NS = 4;
S_Sum = cell(NS,1);
S_Alpha = cell(NS,1);
S_FW = cell(NS,1);
S_GP = cell(NS,1);

%% Increase source rate
for factor = 1:NS
% F-W

[S_Alpha{factor}, lambda_e1] = MaxAlpha_2(TWINDi,TWINDj,TW,V,lambda_s*factor,L,p,Learner,Source,5,t_l);

[S_Sum{factor}, lambda_e2] = MaxAlpha_2(TWINDi,TWINDj,TW,V,lambda_s*factor,L,p,Learner,Source,0,t_l);

lambda_fw = zeros(L,p);
for k = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_fw, T, X, sigma, Sigma, t_prime, n_sample);    

    % Do LP, projection
    vk = LP(TWINDi,TWINDj,TW,V,lambda_s*factor,grad,Learner,Source,t_l);
    
    % Update solution
    lambda_fw = lambda_fw + delta * vk;
    U(lambda_fw, X, sigma, Sigma, T)
    S_FW{factor} = lambda_fw;
end

% Projected Gradient
lambda_pg = S_Alpha{factor};
for kk = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_pg, T, X, sigma, Sigma, t_prime, n_sample);    
    
    % Update solution
    lambda_pg = lambda_pg + delta * grad;
    
    % Projection
    lambda_pg = proj(TWINDi,TWINDj,TW,V,lambda_pg,grad,Learner,Source,lambda_s*factor,t_l);
    U(lambda_pg, X, sigma, Sigma, T)
    S_GP{factor} = lambda_pg;
end

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

[Cap_Alpha{cap}, lambda_e1] = MaxAlpha_2(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,5,t_l);

[Cap_Sum{cap}, lambda_e2] = MaxAlpha_2(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,0,t_l);

lambda_fw = zeros(L,p);
for k = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_fw, T, X, sigma, Sigma, t_prime, n_sample);    

    % Do LP, projection
    vk = LP(TWINDi,TWINDj,TW,V,lambda_s,grad,Learner,Source,t_l);
    
    % Update solution
    lambda_fw = lambda_fw + delta * vk;
    U(lambda_fw, X, sigma, Sigma, T)
    Cap_FW{cap} = lambda_fw;
end

% Projected Gradient
lambda_pg = Cap_Alpha{cap};
for kk = 1:K
    % Estimate the current gradient
    grad = gradient_1(lambda_pg, T, X, sigma, Sigma, t_prime, n_sample);    
    
    % Update solution
    lambda_pg = lambda_pg + delta * grad;
    
    % Projection
    lambda_pg = proj(TWINDi,TWINDj,TW,V,lambda_pg,grad,Learner,Source,lambda_s,t_l);
    U(lambda_pg, X, sigma, Sigma, T)
    Cap_GP{cap} = lambda_pg;
end

end

end