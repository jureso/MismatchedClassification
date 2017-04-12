function [P] = UBmismatch(model, mmodel, sigma2)
%UBMISMATCH Simulation of the upper bound to the error probability
%
% Jure Sokolic, "Mismatch in the Classification of Linear Subspaces:
% Sufficient Conditions for Reliable Classification"
% 
% Copyright @ Jure Sokolic, 2015
% jure.sokolic.13@ucl.ac.uk    

    % get theoretical analysis of the model
    [P_params, pij_params] = analyseAll(model,mmodel);
   
    % ambient dimension
    N = size(model.Sigma(:,:,1),1);
    
    % number of classes
    L = size(model.Sigma,3);
    
    % simulation: ober all noise levels
    for is = 1:length(sigma2) 
        Is2 = sigma2(is)*eye(N);
        for i = 1:L
            sig(:,:,i) = model.Sigma(:,:,i) + Is2;
            sigt(:,:,i) = mmodel.Sigma(:,:,i) + Is2;
        end
        
        P.Pe(is) = 0;
        
        for i = 1:L
            for j = 1:L
                if i == j
                    P.peij{i,j} = [];
                else
                    % set alpha based on theoretically optimal one
                    if isempty(pij_params{i,j}.alpha)
                        alpha = 0.01;
                    else
                        alpha = pij_params{i,j}.alpha;
                    end
                    
                    sig12 = inv(sig(:,:,i)) + alpha * (inv(sigt(:,:,j)) - inv(sigt(:,:,i)));
                    if det(sig12) > 0
                        P.peij{i,j}(is) = (mmodel.P(j)/mmodel.P(i))^alpha * (det(sigt(:,:,i))/det(sigt(:,:,j)))^(alpha/2)*(det(sig(:,:,1))*det(sig12))^(-0.5);
                    else 
                        P.peij{i,j}(is) = 1;
                    end
                    if P.peij{i,j}(is) > 1
                        P.peij{i,j}(is) = 1;
                    end
                    P.Pe(is) = P.Pe(is) + model.P(i)*P.peij{i,j}(is);
                end
            end
        end        
        
    end
    
    % analytically obtained curve curve, measurement gain nor implemented!!!
    for i = 1:L
        for j = 1:L
            if j == i
                P.peij_a{i,j} = [];
            else
                P.peij_a{i,j} = (pij_params{i,j}.g ./ sigma2).^(- pij_params{i,j}.d);
            end
        end
    end
    
    P.Pe_a = (P_params.g ./ sigma2).^(-P_params.d);
    


end

