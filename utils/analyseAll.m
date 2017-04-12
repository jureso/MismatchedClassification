function [P_params, pij_params] = analyseAll(model,mmodel)
%ANALYSEALL Phase transition analysis for all pairs of Gaussians in
%multi-class problem
%
%   P_params ... theoretical analysis of a given classification problem
%   pij_params ... theoretical analysis of all pairs of classes of a given
%   classification problem
%

% Jure Sokolic, "Mismatch in the Classification of Linear Subspaces:
% Sufficient Conditions for Reliable Classification"
% 
% Copyright @ Jure Sokolic, 2015
% jure.sokolic.13@ucl.ac.uk

    % number of classes
    L = size(model.Sigma,3);
    
    % the smallest value of d over all pairs of classes
    dmin = Inf;    
    for i = 1:L
        for j = 1:L
            if j == i
                pij_params{i,j}.cond2 = [];
                pij_params{i,j}.cond1 = [];
                pij_params{i,j}.alpha0 = [];
                pij_params{i,j}.alpha = [];
                pij_params{i,j}.d = [];
                pij_params{i,j}.g = [];
            else
                % extract model parameters
                muij =  model.mu(:,[i,j]);
                mutij =  mmodel.mu(:,[i,j]);
                Pij = model.P([i,j])./sum(model.P([i,j]));
                Ptij = mmodel.P([i,j])./sum(mmodel.P([i,j]));
                sigij(:,:,1) = model.Sigma(:,:,i);
                sigij(:,:,2) = model.Sigma(:,:,j);
                sigtij(:,:,1) = mmodel.Sigma(:,:,i);
                sigtij(:,:,2) = mmodel.Sigma(:,:,j);
                
                modelij.mu = muij;
                modelij.P = Pij;
                modelij.Sigma = sigij;
                
                mmodelij.mu = mutij;
                mmodelij.P = Ptij;
                mmodelij.Sigma = sigtij;
                
                % analyse two classes
                [cond1, cond2, vars] = analyse12(modelij, mmodelij);
                
                % collect parameters
                if cond1 && cond2       
                    pij_params{i,j}.cond2 = cond2;
                    pij_params{i,j}.cond1 = cond1;
                    pij_params{i,j}.alpha0 = vars.alpha0;
                    pij_params{i,j}.alpha = vars.alpha;
                    pij_params{i,j}.d = vars.d;
                    pij_params{i,j}.g = vars.g;
                    
                    d = vars.d;
                    if d < dmin
                        dmin = d;
                    end
                else
                    pij_params{i,j}.cond2 = cond2;
                    pij_params{i,j}.cond1 = cond1;
                    pij_params{i,j}.alpha0 = -1;
                    pij_params{i,j}.alpha = [];
                    pij_params{i,j}.d = 0;
                    pij_params{i,j}.g = 0;
                    
                    dmin = 0;
                end
            end
            
            
        end
    end
    
    % set global parameters
    if dmin > 0
        P_params.d = dmin;
        g = 0;
        for i = 1:L
            for j = 1:L
                if j == i
                else
                    if pij_params{i,j}.d >0.95*dmin && pij_params{i,j}.d <1.05*dmin
                        g = g + model.P(i).*(pij_params{i,j}.g)^(-dmin);
                    end
                end
            end
        end
        P_params.g = (g)^(-1/dmin);
    else
        P_params.d = 0;
        P_params.g = 0;
    end
    % 
end

