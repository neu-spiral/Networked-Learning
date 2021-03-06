function [vk, lambda_e] = MaxAlpha_2(TWINDi,TWINDj,TW,V,lambda_s,L,p,Learner,Source,alpha,t_l)
%LP Solve a LP to find the best direction
%   Detailed explanation goes here

%[L, p] = size(grad);
E = length(TW);
[~,~,Tao] = size(lambda_s); 
S = length(Source);
Nodes = 1:V;
Med = setdiff(Nodes,Source);
Med = setdiff(Med,Learner);
M = length(Med);


cvx_begin quiet
    variable vk(L,p);
    variable lambda_e(E,p,Tao);
    gain=0;
    for l = 1:L
        gain = gain + pow_p(sum(vk(l,:)), 1-alpha)/(1-alpha); 
    end
    maximize(gain);
    subject to
        for e=1:E
            sum(sum(lambda_e(e,:,:))) <= TW(e);
        end
        
        for s=1:S
            out_s = find(TWINDi == Source(s));
            n_s = length(out_s);
            for i=1:p
                for tao = 1:Tao
                    sum_si = 0;
                    for ns = 1:n_s
                        sum_si = sum_si + lambda_e(out_s(ns),i,tao);
                    end
                end
                sum_si <= lambda_s(s,i,tao);
            end
        end
        
        for l=1:L
            in_l = find(TWINDj == Learner(l));
            n_l = length(in_l);
            for i=1:p
                sum_li = 0;
                for nl = 1:n_l
                    sum_li = sum_li + lambda_e(in_l(nl),i,t_l(l));
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
                for tao = 1:Tao
                    sum_in = 0;
                    sum_out = 0;
                    for nmi = 1:n_m_i
                        sum_in = sum_in + lambda_e(in_m(nmi),i,tao);
                    end
                    for nmo = 1:n_m_o
                        sum_out = sum_out + lambda_e(out_m(nmo),i,tao);
                    end
                    sum_out <= sum_in;
                end
            end
        end
        
        lambda_e >= 0;
    
cvx_end

%opt=cvx_optval;

end

