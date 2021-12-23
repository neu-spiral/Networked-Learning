% Star topology
TWINDi = [3 3 4 5 6];
TWINDj = [1 2 3 3 3];
TW = 50 + 100*rand(1,length(TWINDi));
V = 11;

L = 2;
S = 3;
Learner = [1 2];
Source = [4 5 6];

t_l = [1 2];%type indicator of the learners

% Remove the outcoming edges of the learners
for l = 1:L
    findl = find(TWINDi ~= Learner(l));
    TWINDi = TWINDi(findl);
    TWINDj = TWINDj(findl);
    TW = TW(findl);
end