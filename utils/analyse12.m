function [cond1, cond2, vars] = analyse12(model, mmodel)
%ANALYSE12 Determines phase transition for misclassification probability
%of class 1 into class 2
%
% Jure Sokolic, "Mismatch in the Classification of Linear Subspaces:
% Sufficient Conditions for Reliable Classification"
% 
% Copyright @ Jure Sokolic, 2015
% jure.sokolic.13@ucl.ac.uk

    % ambient dimension
    N = size(model.Sigma,1);
    
    % ranks of subspaces
    r1 = rank(model.Sigma(:,:,1));
    r2 = rank(model.Sigma(:,:,2));
    r1t = rank(mmodel.Sigma(:,:,1));
    r2t = rank(mmodel.Sigma(:,:,2));
    
    % basis for the image and positive eigenvalues
    [u1, lambda1] = svds(model.Sigma(:,:,1),r1);
    [u2, lambda2] = svds(model.Sigma(:,:,2),r2);
    [u1t, lambda1t] = svds(mmodel.Sigma(:,:,1),r1t);
    [u2t, lambda2t] = svds(mmodel.Sigma(:,:,2),r2t);
    
    % find intersection of u1t and u2t
    [h, c, j] = svd(u1t'*u2t);
    if size(c,2) > 1;
        c = diag(c);
    end
    idx = find(c<1-1e-6);
    if isempty(idx)
        idx = r1t+1;
    end
    u1th = u1t*h;
    u2tj = u2t*j;
    u12t = u1th(:,1:idx-1);
    
    % determine u1tp and ut1tp
    u1tp = u1th(:,idx:end);
    u2tp = u2tj(:,idx:end);
    
    % ker(u1tp') and ker(u2tp')
    k1tp = null(u1tp');
    k2tp = null(u2tp');
    
    % intersection of im(u1) and ker(u1tp') and the complement which
    % leads to v1 and w1
    [h, c, j] = svd(u1'*k1tp);
    if size(c,2) > 1;
        c = diag(c);
    end
    idx = find(c<1-1e-6);
    if isempty(idx)
        idx = r1t+1;
    end
    u1h = u1*h;
    w1 = u1h(:,1:idx-1);
    v1 = u1h(:,idx:end);
    
    % sufficient conditions: im(w1) \subseteq ker(u2tp')
    if isempty(w1)
        cond2 = 1;
    else
        if min(svd(w1'*k2tp)) < 1-1e-3 % offset added due to numerical reasons
            cond2 = 0;
        else
            cond2 = 1;
        end
    end
    
    % sufficinet conditions: r_11 > 0 or v1'*(u1t*u1t' - u2t*u2t')*v1 > 0
    r11 = rank(v1);
    if r11 == 0
        cond1 = 1;
        alpha120 = min([1, min(diag(lambda1t))/(1+ max(diag(lambda1))), min(diag(lambda1t))/(1+ min(diag(lambda1t)))]);
    else
        ev = eig(v1'*(u1t*u1t' - u2t*u2t')*v1);
        c0 = min(ev);
        c1 = max(ev);
        if c0 > -1e-3*abs(c1)% offset added due to numerical reasons
            cond1 = 1;
            % maximum possible alpha
            alpha120 = min([1, min(diag(lambda1t))/(1+ max(diag(lambda1))), min(diag(lambda1t))/(1+ min(diag(lambda1t))), c0/(1+c0*(1+1/min(diag(lambda1t))))]);
        else
            cond1 = 0;
            alpha120 = -1;
        end
    end
    if cond2 && cond1       
        vars.alpha0 = alpha120;
        if (r2t-r1t) > 0
            vars.alpha = alpha120-1e-6;
        else
            vars.alpha = 1e-6;
        end
        vars.d = 0.5*(r11 + vars.alpha*(r2t-r1t));
        % computation of measurement gain not implemented
        vars.g = 1;
              
    else
        vars.alpha0 = -1;
    end

end

