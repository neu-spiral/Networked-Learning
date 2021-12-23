% Grid topology
TWINDi = [1 1 2 2 2 3 3 3 4 4 5 5 5 6 6 6 6  7 7 7 7  8 8 8  9 9  9  10 10 10 10 11 11 11 11 12 12 12 13 13 14 14 14 15 15 15 16 16];
TWINDj = [2 5 1 3 6 2 4 7 3 8 1 6 9 2 5 7 10 3 6 8 11 4 7 12 5 10 13 6  9  11 14 7  10 12 15 8  11 16 9  14 10 13 15 11 14 16 12 15];
TW = 50 + 100*rand(1,length(TWINDi));
V=16;

L = 2;
S = 3;
Learner = [1 5];
Source = [8 12 16];

t_l = [1 2];%type indicator of the learners

% Remove the outcoming edges of the learners
for l = 1:L
    findl = find(TWINDi ~= Learner(l));
    TWINDi = TWINDi(findl);
    TWINDj = TWINDj(findl);
    TW = TW(findl);
end