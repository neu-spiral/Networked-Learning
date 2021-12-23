%% config
%1.topology
topSel = 'GRID';%value = {'GEANT','Abilene','FatTree','GRID','Star'}

T=1;%simulation length
p = 20;%number of data features
d = 100;%feature dimension

Tao = 2;%number of learner types
%t_l = [1 2];%move this to topology files

K = 100;%number of iteration for F-W
delta = 1/K;%step size for F-W
n_sample = 30;
t_prime = 15;

%% generating experiments
if strcmpi(topSel,'GEANT')
    GEANT;
elseif strcmpi(topSel,'Abilene')
    Abilene;
elseif strcmpi(topSel,'FatTree')
    FatTree;
elseif strcmpi(topSel,'GRID')
    GRID;
elseif strcmpi(topSel,'Star')
    Star;
end

% All features
%X = abs(randn([p,d]));
X = rand([p,d]);
for i = 1:p
    for j = 1:d
        if X(i,j)>0.3
            X(i,j) = 10 + 5*rand;
        end
    end
end

sigma = rand;

% Source rates, need to consider different types
lambda_s = zeros(S,p,Tao);
for tao = 1:Tao
    lambda_s(:,:,tao) = 3*rand([S,p]);
end

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

%% run main function
[Cap_Sum,Cap_Alpha,Cap_FW,Cap_GP,S_Sum,S_Alpha,S_FW,S_GP] = main_simulation(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,t_l, T, X, sigma, Sigma, t_prime, n_sample,K,delta);