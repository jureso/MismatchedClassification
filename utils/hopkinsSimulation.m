function [results] = hopkinsSimulation(X, r, trainsamples, nruns,cov_regularization,n1,n2,n3)
%HOPKINSSIMULATION Simulate hopkins true error phase transition and upper
%bound phase transition
%
% Jure Sokolic, "Mismatch in the Classification of Linear Subspaces:
% Sufficient Conditions for Reliable Classification"
% 
% Copyright @ Jure Sokolic, 2015
% jure.sokolic.13@ucl.ac.uk
    
    % number of classes is assumed to be 3!!!
    classes = 3;
    % ambient dimension
    N = size(X{1},1);
    
    % number of samples and "true model"
    for i = 1:classes
        nsamples{i} = size(X{i},2);
        label{i} = i*ones(1,nsamples{i}); 
        ntrain{i} = trainsamples;
        ntest{i} = nsamples{i} - trainsamples;
        
        cMat = X{i}*X{i}'./nsamples{i};
        [u_ s_ v_] = svd(cMat);
        s_diag = diag(s_);
        u{i} = u_(:,1:r);
        s{i} = s_diag(1:r);
        s_(r+1:end,r+1:end) = 0;
        sigma2 = mean(s_diag(r+1:end).^2);

        sig(:,:,i) = u_*s_*v_' + sigma2*eye(N) + cov_regularization*eye(N); 
        sig(:,:,i) = (sig(:,:,i) + sig(:,:,i)')./2; % force matrices to be symmetric

    end
    labels = cell2mat(label);
    
    % true model error
    gmmTrue = gmdistribution(zeros(classes,N), sig);
    chat = cluster(gmmTrue, cell2mat(X)');
    errTrue = mean(labels~=chat');
    results.errTrue = errTrue;
    
    % simulate training
    parfor ir = 1:nruns
        [errMismatch{ir}, boundPT{ir}] = run_experiment(X,u, s, nsamples, ntrain, ntest, classes,r, sigma2,cov_regularization, N,n1,n2,n3)
        fprintf('Finished run: %d/%d\n', ir,nruns)
    end
    results.errMismatch = errMismatch;
    results.boundPT = boundPT;
end

function [errMismatch, boundPT] = run_experiment(X,u, s, nsamples, ntrain, ntest, classes,r, sigma2,cov_regularization, N,n1,n2,n3)
    % randomly split traning testing data
    for i = 1:classes
        idx = randsample(nsamples{i},nsamples{i});
        Xtrain{i} = X{i}(:,idx(1:ntrain{i}));
        Xtest{i} = X{i}(:,idx(ntrain{i}+1:end));
        C{i} = i*ones(1,size(Xtrain{i},2));
        Ctest{i} = i*ones(1,size(Xtest{i},2));
    end
    C = cell2mat(C);
    Ctest = cell2mat(Ctest);
    Xtest =cell2mat(Xtest);
    
    % set number of training samples n_1
    for i1 = 1:length(n1)
        % training samples
        Xtrain_{1} = Xtrain{1}(:,1:n1(i1));
        
        % estimated model of class 1
        i = 1;
        cMat = Xtrain_{i}*Xtrain_{i}'./size(Xtrain_{i},2);
        [u_ s_ v_] = svd(cMat);
        ut{i} = u_(:,1:r);
        sigt(:,:,i) = u_*s_*v_' + sigma2*eye(N) + cov_regularization*eye(N); 
        sigt(:,:,i) = (sigt(:,:,i) + sigt(:,:,i)')./2; % force matrices to be symmetric
        
        % set number of training samples n_2
        for i2 = 1:length(n2)
            % training samples
            Xtrain_{2} = Xtrain{2}(:,1:n2(i2));
            
            % estimated model of class 2
            i = 2;
            cMat = Xtrain_{i}*Xtrain_{i}'./size(Xtrain_{i},2);
            [u_ s_ v_] = svd(cMat);
            ut{i} = u_(:,1:r);
            sigt(:,:,i) = u_*s_*v_' + sigma2*eye(N) + cov_regularization*eye(N); 
            sigt(:,:,i) = (sigt(:,:,i) + sigt(:,:,i)')./2; % force matrices to be symmetric
            
            % set number of training samples n_3
            for i3 = 1:length(n3)
                % training samples
                Xtrain_{3} = Xtrain{3}(:,1:n3(i3));
            
                % estimated model of class 3
                i = 3;
                cMat = Xtrain_{i}*Xtrain_{i}'./size(Xtrain_{i},2);
                [u_ s_ v_] = svd(cMat);
                ut{i} = u_(:,1:r);
                sigt(:,:,i) = u_*s_*v_' + sigma2*eye(N) + cov_regularization*eye(N); 
                sigt(:,:,i) = (sigt(:,:,i) + sigt(:,:,i)')./2; % force matrices to be symmetric
            
            
                % classify test samples
                gmmt = gmdistribution(zeros(classes,N), sigt);
                chat = cluster(gmmt, Xtest');      
                errMismatch(i1,i2,i3) = mean(Ctest~=chat');
                
                % theoretical predicion
                boundPT(i1,i2,i3) = phasetransition(u,ut);
            end
        end
        
    end
end

function d = phasetransition(u,ut)
    d = 0; % phase transition variable
    % iteration over all pairs is handcoded
    % all pairs 12, 21, 13, 31, 23, 32
    idx1 = [1, 2, 1, 3, 2, 3];
    idx2 = [2, 1, 3, 1, 3, 2];
    
    % iterate over all pair of classes
    for i = 1:length(idx1);
        % pairwise phase transitions
        di(i) = phasetransition12(u{idx1(i)}, ut{idx1(i)}, ut{idx2(i)});
    end
    
    % no error floor for all classes
    if min(di) > 0
        d = 1;
    else
        d = 0;
    end

end

function d = phasetransition12(u1,ut1,ut2)
    % subspace dimension
    r= size(u1,2);
    % ambient dimensions
    N = size(u1,1);
        
    % find intersection of u1t and u2t
    [h, c, j] = svd(ut1'*ut2);
    if size(c,2) > 1;
        c = diag(c);
    end
    idx = find(c<1-1e-6);
    if isempty(idx)
        idx = r+1;
    end
    
    ut1h = ut1*h;
    ut2j = ut2*j;
    ut12 = ut1h(:,1:idx-1);
    
    % determine u1tp and ut1tp
    ut1p = ut1h(:,idx:end);
    ut2p = ut2j(:,idx:end);
    
    % ker(u1tp') and ker(u2tp')
    kt1p = null(ut1p');
    kt2p = null(ut2p');
    
    % intersection of im(u1) and ker(u1tp') and the complement which
    % leads to v1 and w1
    [h, c, j] = svd(u1'*kt1p);
    if size(c,2) > 1;
        c = diag(c);
    end
    idx = find(c<1-1e-6);
    if isempty(idx)
        idx = r+1;
    end
    u1h = u1*h;
    w1 = u1h(:,1:idx-1);
    v1 = u1h(:,idx:end);
    
    % sufficient conditions: im(w1) \subseteq ker(u2tp')
    if isempty(w1)
        cond2 = 1;
    else
        if min(svd(w1'*kt2p)) < 1-1e-3 % offset added due to numerical reasons
            cond2 = 0;
        else
            cond2 = 1;
        end
    end
    
    % sufficinet conditions: r_11 > 0 or v1'*(u1t*u1t' - u2t*u2t')*v1 > 0
    r11 = rank(v1);
    if r11 == 0
        cond1 = 1;
    else
        ev = eig(v1'*(ut1*ut1' - ut2*ut2')*v1);
        c0 = min(ev);
        c1 = max(ev);
        if c0 > -1e-3*abs(c1) % offset added due to numerical reasons
            cond1 = 1;
        else
            cond1 = 0;
        end
    end
    
    if r11>0 && cond1 > 0 && cond2 > 0
        d = 1;
    else
        d = 0;
    end
        
end
