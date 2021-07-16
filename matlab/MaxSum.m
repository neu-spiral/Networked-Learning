function vk = MaxSum(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source)
%LP Solve a LP to find the best direction
%   Detailed explanation goes here

%[L, p] = size(grad);
E = length(TW);
S = length(Source);
Nodes = 1:V;
Med = setdiff(Nodes,Source);
Med = setdiff(Med,Learner);
M = length(Med);


cvx_begin quiet
    variable vk(L,p);
    variable lambda_e(E,p);
    gain=sum(sum(vk));
            
    maximize(gain);
    subject to
        for e=1:E
            sum(lambda_e(e,:)) <= TW(e);
        end
        
        for s=1:S
            out_s = find(TWINDi == Source(s));
            n_s = length(out_s);
            for i=1:p
                sum_si = 0;
                for ns = 1:n_s
                    sum_si = sum_si + lambda_e(out_s(ns),i);
                end
                sum_si <= lambda_s(s,i);
            end
        end
        
        for l=1:L
            in_l = find(TWINDj == Learner(l));
            n_l = length(in_l);
            for i=1:p
                sum_li = 0;
                for nl = 1:n_l
                    sum_li = sum_li + lambda_e(in_l(nl),i);
                end
                vk(l,i) == sum_li; 
            end
        end
        
        for m = 1:M
            out_m = find(TWINDi == Med(m));
            n_m_o = length(out_m);
            in_m = find(TWINDj == Med(m));
            n_m_i = length(in_m);
            for i=1:p
               sum_in = 0;
               sum_out = 0;
               for nmi = 1:n_m_i
                   sum_in = sum_in + lambda_e(in_m(nmi),i);
               end
               for nmo = 1:n_m_o
                   sum_out = sum_out + lambda_e(out_m(nmo),i);
               end
               sum_out <= sum_in;
            end
        end
        
        lambda_e >= 0;
    
cvx_end

%opt=cvx_optval;

end

