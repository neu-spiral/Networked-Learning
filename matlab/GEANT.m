% GEANT topology
TWINDi = [1 1 1  1  1  2 2 3 3 3 3  4 4 4  4  5 5  6 6 7 7 7 8 8 9 9  9  10 10 11 11 12 12 12 12 12 13 13 14 14 14 14 15 15 15 15 15 16 16 17 17 17 17 18 18 18 18 19 19 20 20 20 21 21 22 22];
TWINDj = [2 3 15 17 22 1 3 1 2 4 12 3 7 15 18 6 14 5 7 4 6 8 7 9 8 10 12 9  11 10 12 3  9  11 13 15 12 14 5  13 15 16 1  4  12 14 17 14 17 1  15 16 18 4  17 19 20 18 20 18 19 21 20 22 1  21];
TW = 50 + 100*rand(1,length(TWINDi));
V=22;

L = 2;
S = 3;
Learner = [1 5];
Source = [10 15 20];

t_l = [1 2];%type indicator of the learners

% Remove the outcoming edges of the learners
for l = 1:L
    findl = find(TWINDi ~= Learner(l));
    TWINDi = TWINDi(findl);
    TWINDj = TWINDj(findl);
    TW = TW(findl);
end