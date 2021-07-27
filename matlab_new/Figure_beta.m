% U(lambda_fw, X, sigma, Sigma, T)
% 
% U(lambda_fwl, X, sigma, Sigma, T)
% 
% U(lambda_pg, X, sigma, Sigma, T)
% 
% U(v1, X, sigma, Sigma, T)
% 
% U(v2, X, sigma, Sigma, T)
% 
% U(0, X, sigma, Sigma, T)

beta = randn(d,1);

beta1 = cell(l,NC);
beta2 = cell(l,NC);
beta3 = cell(l,NC);

CapVector = 1:4;
SumVector = zeros(1,4);
AlphaVector = zeros(1,4);
FWVector = zeros(1,4);
PGVector = zeros(1,4);



% % Figure for capacity beta
% for i=1:4
%     for j = 1: 10000
%     
%     SumVector(i) = SumVector(i) + norm(beta - Beta(Cap_Sum{i}, X, sigma, Sigma, T, beta, 2, 1)) + norm(beta - Beta(Cap_Sum{i}, X, sigma, Sigma, T, beta, 2, 2));
%     AlphaVector(i) = AlphaVector(i) + norm(beta - Beta(Cap_Alpha{i}, X, sigma, Sigma, T, beta, 2, 1)) + norm(beta - Beta(Cap_Alpha{i}, X, sigma, Sigma, T, beta, 2, 2));
%     FWVector(i) = FWVector(i) + norm(beta - Beta(Cap_FW{i}, X, sigma, Sigma, T, beta, 2, 1)) + norm(beta - Beta(Cap_FW{i}, X, sigma, Sigma, T, beta, 2, 2));
%     PGVector(i) = PGVector(i) + norm(beta - Beta(Cap_GP{i}, X, sigma, Sigma, T, beta, 2, 1)) + norm(beta - Beta(Cap_GP{i}, X, sigma, Sigma, T, beta, 2, 2));
%     
%     end
% end

% Figure for capacity beta
for i=1:4
    for j = 1: 10000
    
    SumVector(i) = SumVector(i) + norm(beta - Beta(Cap_Sum{i}, X, sigma, Sigma, T, beta, 2, 1));
    AlphaVector(i) = AlphaVector(i) + norm(beta - Beta(Cap_Alpha{i}, X, sigma, Sigma, T, beta, 2, 1));
    FWVector(i) = FWVector(i) + norm(beta - Beta(Cap_FW{i}, X, sigma, Sigma, T, beta, 2, 1));
    PGVector(i) = PGVector(i) + norm(beta - Beta(Cap_GP{i}, X, sigma, Sigma, T, beta, 2, 1));
    
    end
end

SumVector = SumVector/10000;
AlphaVector = AlphaVector/10000;
FWVector = FWVector/10000;
PGVector = PGVector/10000;

figure
hold on;grid on;box on
% plot(CapVector, SumVector);
% plot(CapVector, AlphaVector);
% plot(CapVector, FWVector);
% plot(CapVector, PGVector);
plot(CapVector,SumVector,'-h','LineWidth',4,'MarkerSize',20,'Color',1/255*[204, 0, 150])
plot(CapVector,AlphaVector,'-s','LineWidth',4,'MarkerSize',20,'Color',1/255*[200, 132, 0])
plot(CapVector,FWVector,'--s','LineWidth',4,'MarkerSize',20,'Color',1/255*[0, 132, 150])
plot(CapVector,PGVector,'-p','LineWidth',4,'MarkerSize',20,'Color',1/255*[100, 132, 100])

h=legend('Sum','Alpha','FW','PG');

xlabel('Link Capacity Downsize Factor','FontSize',18)
%ylabel('Distance between MAP Estimator and True Model','FontSize',18)
ylabel('Norm of Estimation Error (Average)','FontSize',18)

xticks(CapVector)
% ylim([0 max(Cost_WCS_avg)])
set(h,'FontSize',16);
set(gca,'FontSize',16)

% CapVector = 1:4;
% SumVector = zeros(1,4);
% AlphaVector = zeros(1,4);
% FWVector = zeros(1,4);
% PGVector = zeros(1,4);
% 
% Figure for source beta
for i=1:4
    for j = 1: 10000
    
    SumVector(i) = SumVector(i) + norm(beta - Beta(S_Sum{i}, X, sigma, Sigma, T, beta, 2, 1));
    AlphaVector(i) = AlphaVector(i) + norm(beta - Beta(S_Alpha{i}, X, sigma, Sigma, T, beta, 2, 1));
    FWVector(i) = FWVector(i) + norm(beta - Beta(S_FW{i}, X, sigma, Sigma, T, beta, 2, 1));
    PGVector(i) = PGVector(i) + norm(beta - Beta(S_GP{i}, X, sigma, Sigma, T, beta, 2, 1));
    
    end
end

SumVector = SumVector/10000;
AlphaVector = AlphaVector/10000;
FWVector = FWVector/10000;
PGVector = PGVector/10000;

figure
hold on;grid on;box on
% plot(CapVector, SumVector);
% plot(CapVector, AlphaVector);
% plot(CapVector, FWVector);
% plot(CapVector, PGVector);
plot(CapVector,SumVector,'-h','LineWidth',4,'MarkerSize',20,'Color',1/255*[204, 0, 150])
plot(CapVector,AlphaVector,'-s','LineWidth',4,'MarkerSize',20,'Color',1/255*[200, 132, 0])
plot(CapVector,FWVector,'--s','LineWidth',4,'MarkerSize',20,'Color',1/255*[0, 132, 150])
plot(CapVector,PGVector,'-p','LineWidth',4,'MarkerSize',20,'Color',1/255*[100, 132, 100])

h=legend('Sum','Alpha','FW','PG');

xlabel('Source Rate Scaling Factor','FontSize',18)
%ylabel('Distance between MAP Estimator and True Model','FontSize',18)
ylabel('Norm of Estimation Error (Average)','FontSize',18)

xticks(CapVector)
% ylim([0 max(Cost_WCS_avg)])
set(h,'FontSize',16);
set(gca,'FontSize',16)