% We learn for several periods and update the prior...

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
lambda_s = 10*rand([S,p]);
sigma = rand;
%Sigma = diag(rand(d,1));
Sigma = cell(L,1);
for l = 1:L
    Sigma{l} = 2*diag(abs(randn(d,1)));
end

n_period = 4;

Sigma1 = cell(L,1);
for l = 1:L
    Sigma1{l} = diag(rand(d,1));
end
Sigma2 = Sigma1;
Sigma3 = Sigma1;

K = 100;
delta = 1/K;
n_sample = 30;
t_prime = 10;

beta = randn(d,1);

beta1 = cell(l,n_period);
beta2 = cell(l,n_period);
beta3 = cell(l,n_period);

[v1, lambda_e1] = MaxAlpha(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,10);

[v2, lambda_e2] = MaxAlpha(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,0);

tic
for nn = 1:n_period
    nn
    
    lambda_fw = 0.001*ones(L,p);
    lambda_fwl = lambda_fw;

% F-W
    for k = 1:K
        % Estimate the current gradient
        grad = gradient_1(lambda_fw, T, X, sigma, Sigma1, t_prime, n_sample);    

        % Do LP, projection
        vk = LP(TWINDi,TWINDj,TW,V,lambda_s,grad,Learner,Source);
    
        % Update solution
        lambda_fw = lambda_fw + delta * vk;
        
        k
    end
    % Decide the number of received data (random)
    N = poissrnd(lambda_fw*T);
    
    for l = 1:L
        % MAP estimator
        X_M = zeros(sum(N(l,:)),d);
        count = 0;
        for i = 1:p
            for j = 1:N(l,i)
                count = count + 1;
                X_M(count,:) = X(i,:);
            end
        end 
        beta1{l,nn} = (X_M' * X_M + sigma^2 * Sigma1{l}^-1)^-1 * (X_M') * (X_M * beta + normrnd(0,sigma,[sum(N(l,:)),1])); 
        
        % Update prior
        Sigma1{l} = diag(diag(Sigma1{l} * (X_M') * X_M * ((X_M') * X_M + sigma^2 * Sigma1{l}^-1)^-1));

    end
    
    %save('PriorTest.mat')

% % F-W with lesser samples
%     for k = 1:K
%         % Estimate the current gradient
%         grad = gradient_1(lambda_fwl, T, X, sigma, Sigma2, t_prime, 10);    
% 
%         % Do LP, projection
%         vk = LP(TWINDi,TWINDj,TW,V,lambda_s,grad,Learner,Source);
%     
%         % Update solution
%         lambda_fwl = lambda_fwl + delta * vk;
%     end
%     % Decide the number of received data (random)
%     N = poissrnd(lambda_fw*T);    
%     for l = 1:L
%         % MAP estimator
%         X_M = zeros(sum(N(l,:)),d);
%         count = 0;
%         for i = 1:p
%             for j = 1:N(l,i)
%                 count = count + 1;
%                 X_M(count,:) = X(i,:);
%             end
%         end 
%         beta2{l,nn} = (X_M' * X_M + sigma^2 * Sigma2{l}^-1)^-1 * (X_M') * (X_M * beta + normrnd(0,sigma,[sum(N(l,:)),1])); 
%         
%         % Update prior
%         Sigma2{l} = Sigma2{l} * (X_M') * X_M * ((X_M') * X_M + sigma^2 * Sigma2{l}^-1)^-1;
% 
%     end
   
    
% Projected Gradient
    lambda_pg = v2;
    for kk = 1:K
        % Estimate the current gradient
        grad = gradient_1(lambda_pg, T, X, sigma, Sigma3, t_prime, n_sample);    
    
        % Update solution
        lambda_pg = lambda_pg + delta * grad;
    
        % Projection
        lambda_pg = proj(TWINDi,TWINDj,TW,V,lambda_pg,grad,Learner,Source,lambda_s);
    end
    % Decide the number of received data (random)
    N = poissrnd(lambda_fw*T);
    for l = 1:L
        % MAP estimator
        X_M = zeros(sum(N(l,:)),d);
        count = 0;
        for i = 1:p
            for j = 1:N(l,i)
                count = count + 1;
                X_M(count,:) = X(i,:);
            end
        end 
        beta3{l,nn} = (X_M' * X_M + sigma^2 * Sigma3{l}^-1)^-1 * (X_M') * (X_M * beta + normrnd(0,sigma,[sum(N(l,:)),1])); 
        
        % Update prior
        Sigma3{l} = Sigma3{l} * (X_M') * X_M * ((X_M') * X_M + sigma^2 * Sigma3{l}^-1)^-1;

    end
    
    toc
end

save('PriorUpdate.mat')