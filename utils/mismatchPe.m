function [P, Pm] = mismatchPe(model, mmodel, param)
%SIMULATE MC simulation of the misclassification probability
%   Number of classes is assumed to be 2. Measurement matrix not
%   implemented yet.
%   Outputs:
%       Pm.peij ... misclassification probability of class i as class j
%       Pm.pei ... misclassification probability of class i
%       Pm.Pe ... misclassification probability
%       P.peij ... misclassification probability of class i as class j
%       P.pei ... misclassification probability of class i
%       P.Pe ... misclassification probability 
%
% Jure Sokolic, "Mismatch in the Classification of Linear Subspaces:
% Sufficient Conditions for Reliable Classification"
% 
% Copyright @ Jure Sokolic, 2015
% jure.sokolic.13@ucl.ac.uk

    % number of classes
    L = size(model.Sigma,3);
    % ambient dimension
    N = size(model.Sigma,1);
    
 
    ls2 = length(param.sigma2);
    lr = param.runs;
    
    Pm.Pe = zeros(1,ls2);
    P.Pe = zeros(1,ls2);
    for i = 1:L
        Pm.pei{i} = zeros(1,ls2);
        P.pei{i} = zeros(1,ls2);

        for j = 1:L
            if j == i
                Pm.peij{i,j} = [];
                P.peij{i,j} = [];
            else
                Pm.peij{i,j} = zeros(1,ls2);
                P.peij{i,j} = zeros(1,ls2);
            end
        end
    end    
    
    parfor ir = 1:lr
        experiment{ir}.mPe = zeros(1,ls2);
        experiment{ir}.Pe = zeros(1,ls2);
        for i = 1:L
            experiment{ir}.mpei{i} = zeros(1,ls2);
            experiment{ir}.pei{i} = zeros(1,ls2);

            for j = 1:L
                if j == i
                    experiment{ir}.mpeij{i,j} = [];
                    experiment{ir}.peij{i,j} = [];
                else
                    experiment{ir}.mpeij{i,j} = zeros(1,ls2);
                    experiment{ir}.peij{i,j} = zeros(1,ls2);
                end
            end
        end
    end
    
  
    
    gmmTrue = gmdistribution(model.mu', model.Sigma, model.P);
    
    for ir = 1:param.runs
        [X, class] = random(gmmTrue, param.testsamples);
        X = X';
        
        for is = 1:ls2
            Is2 = param.sigma2(is)*eye(N); 
            Xn = X + sqrt(param.sigma2(is)).*randn(size(X));
            
            sig = []; sigt = [];
            for i = 1:L
                sig(:,:,i) = model.Sigma(:,:,i) + Is2;
                sigt(:,:,i) = mmodel.Sigma(:,:,i) + param.sigma2factor.*Is2;
            end
          
            gmmTrueNoise = gmdistribution(model.mu', sig, model.P);
            gmmMismatchedNoise = gmdistribution(mmodel.mu', sigt, mmodel.P);
            
            cTrue = cluster(gmmTrueNoise, Xn');
            cMismatch = cluster(gmmMismatchedNoise, Xn');
            
            % check this computations and make sure they are correct
            
            experiment{ir}.mPe(is) = mean(class~=cMismatch);
            experiment{ir}.Pe(is) = mean(class~=cTrue);
            for i = 1:L
                experiment{ir}.mpei{i}(is) = mean(cMismatch(class == i)~= i);
                experiment{ir}.pei{i}(is) = mean(cTrue(class == i)~= i);

                for j = 1:L
                    if j == i
                        experiment{ir}.mpeij{i,j} = [];
                        experiment{ir}.peij{i,j} = [];
                    else
                        experiment{ir}.mpeij{i,j}(is) = mean(cMismatch(class == i)== j);
                        experiment{ir}.peij{i,j}(is) = mean(cTrue(class == i)== j);
                    end
                end
            end         
        end
    end
        
for ir = 1:lr
    Pm.Pe = Pm.Pe + experiment{ir}.mPe;
    P.Pe = P.Pe + experiment{ir}.Pe;
    for i = 1:L
        Pm.pei{i} = Pm.pei{i} + experiment{ir}.mpei{i};
        P.pei{i} = P.pei{i} + experiment{ir}.pei{i};

        for j = 1:L
            if j == i
                Pm.peij{i,j} = [];
                P.peij{i,j} = [];
            else
                Pm.peij{i,j} = Pm.peij{i,j} + experiment{ir}.mpeij{i,j};
                P.peij{i,j} = P.peij{i,j} + experiment{ir}.peij{i,j};
            end
        end
    end 
end
    
Pm.Pe = Pm.Pe./lr;
P.Pe = P.Pe./lr;
for i = 1:L
    Pm.pei{i} = Pm.pei{i}./lr;
    P.pei{i} = P.pei{i}./lr;

    for j = 1:L
        if j == i
            Pm.peij{i,j} = [];
            P.peij{i,j} = [];
        else
            Pm.peij{i,j} = Pm.peij{i,j}./lr;
            P.peij{i,j} = P.peij{i,j}./lr;
        end
    end
end 


end



