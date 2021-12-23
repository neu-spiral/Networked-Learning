

beta1 = cell(l,NC);
beta2 = cell(l,NC);
beta3 = cell(l,NC);

CapVector = 1:4;
SumVector = zeros(1,4);
AlphaVector = zeros(1,4);
FWVector = zeros(1,4);
PGVector = zeros(1,4);


% Figure for capacity utility
for i=1:4
    SumVector(i) = U(Cap_Sum{i}, X, sigma, Sigma, T);
    AlphaVector(i) = U(Cap_Alpha{i}, X, sigma, Sigma, T);
    FWVector(i) = U(Cap_FW{i}, X, sigma, Sigma, T);
    PGVector(i) = U(Cap_GP{i}, X, sigma, Sigma, T);
end

figure
hold on;grid on;box on
plot(CapVector,FWVector,'--s','LineWidth',4,'MarkerSize',15,'Color',1/255*[0, 132, 150])
plot(CapVector,SumVector,'-h','LineWidth',4,'MarkerSize',15,'Color',1/255*[200, 132, 0])
plot(CapVector,AlphaVector,'-s','LineWidth',4,'MarkerSize',15,'Color',1/255*[28, 166, 23])
plot(CapVector,PGVector,'--p','LineWidth',4,'MarkerSize',15,'Color',1/255*[225, 50, 0])

h=legend('FW','MaxSum','MaxAlpha','PGA');

xlabel('Link Capacity Downsize Factor','FontSize',18)
ylabel('Aggregate Utility','FontSize',18)

xticks(CapVector)
% ylim([0 max(Cost_WCS_avg)])
set(h,'FontSize',16);
set(gca,'FontSize',16)

for i=1:4
    SumVector(i) = U(S_Sum{i}, X, sigma, Sigma, T);
    AlphaVector(i) = U(S_Alpha{i}, X, sigma, Sigma, T);
    FWVector(i) = U(S_FW{i}, X, sigma, Sigma, T);
    PGVector(i) = U(S_GP{i}, X, sigma, Sigma, T);
end

figure
hold on;grid on;box on
plot(CapVector,FWVector,'--s','LineWidth',4,'MarkerSize',15,'Color',1/255*[0, 132, 150])
plot(CapVector,SumVector,'-h','LineWidth',4,'MarkerSize',15,'Color',1/255*[200, 132, 0])
plot(CapVector,AlphaVector,'-s','LineWidth',4,'MarkerSize',15,'Color',1/255*[28, 166, 23])
plot(CapVector,PGVector,'--p','LineWidth',4,'MarkerSize',15,'Color',1/255*[225, 50, 0])

h=legend('FW','MaxSum','MaxAlpha','PGA');

xlabel('Source Rate Scaling Factor','FontSize',18)
ylabel('Aggregate Utility','FontSize',18)

xticks(CapVector)
% ylim([0 max(Cost_WCS_avg)])
set(h,'FontSize',16);
set(gca,'FontSize',16)