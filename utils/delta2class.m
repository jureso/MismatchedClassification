function [delta] = delta2class(model, mmodel, sigma2)
%DELTA2CLASS Example of  f-Divergence bound for 2 Gaussian classes based on 
% R. Schluter, M. Nussbaum-Thom, E. Beck, T. Alkhouli, and H. Ney,
% “Novel Tight Classification Error Bounds under Mismatch Conditions
%  based on $f$-Divergence,” in Inf. Theory Work. (ITW), 2013 IEEE.
%  IEEE, Sep. 2013, pp. 1–5.
%
% Jure Sokolic, "Mismatch in the Classification of Linear Subspaces:
% Sufficient Conditions for Reliable Classification"
% 
% Copyright @ Jure Sokolic, 2015
% jure.sokolic.13@ucl.ac.uk

    % ambient dimension
    N = size(model.Sigma(:,:,1),1);
    % number of classes: note that this implementataion only supports 2
    % classes
    L = size(model.Sigma,3);
    assert(L == 2,'This function implemented only for a two class case')
    
    % value of difference between bayesian error probability and mismatched
    % error probability
    delta = zeros(1,length(sigma2));
    for is = 1:length(sigma2) 
        Is2 = sigma2(is)*eye(N);
        for i = 1:L
            sig(:,:,i) = model.Sigma(:,:,i) + Is2;
            sigt(:,:,i) = mmodel.Sigma(:,:,i) + Is2;
            isig(:,:,i) = inv(model.Sigma(:,:,i) + Is2);
            
        end
        
        delta(is) = 0;
        
        for i = 1:L
            % kl divergence for zeromean gaussians
            Dkl{i} = trace(isig(:,:,i)*sigt(:,:,i)) - log(det(sigt(:,:,i))/det(sig(:,:,i))) - N;
        end
        
        delta(is) = 0.25.*(Dkl{1}+Dkl{2});
    end
    
end

