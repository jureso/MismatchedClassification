% Jure Sokolic, "Mismatch in the Classification of Linear Subspaces:
% Sufficient Conditions for Reliable Classification"
% 
% Copyright @ Jure Sokolic, 2015
% jure.sokolic.13@ucl.ac.uk
clear; close all; clc
addpath utils/

%% Figure 3
clear; close all; clc
rng(0,'twister') 


N = 2;
model.P = [0.5, 0.5]; % priors
model.mu = [0 0; 0 0]'; % means
mmodel.P = [0.5, 0.5]; % priors
mmodel.mu = [0 0; 0 0]'; % means

param.runs = 8; % repeat the experiments
param.testsamples = 1e5;% number of testing samples per experiment
param.sigma2 = logspace(1,-10,12); % nose variance
param.sigma2factor = 1; % factor of over/under estimation of noise variance

% example a
idx = 1;
% true model parameters
sig(:,:,1) = [1 0; 0 0];
sig(:,:,2) = [0 0; 0 1];
model.Sigma = sig; % covariance matrices
% mismatched parameters
sigt(:,:,1) = [1 0; 0 1];
sigt(:,:,2) = [0 0 ; 0 1];
mmodel.Sigma = sigt; % covariance matrices


% simulate error probability
[P{idx}, Pm{idx}] = mismatchPe(model,mmodel,param);
% simulate upper bound
Pmub{idx} =UBmismatch(model, mmodel, param.sigma2);
% simulate divergence bound
delta{idx} = delta2class(model, mmodel, param.sigma2);
Pubkl{idx} = min(Pm{idx}.Pe+delta{idx}, 1);
% theoretical predictions
[P_params pij_params] = analyseAll(model,mmodel);
d{idx} = P_params.d;


% example b
idx = 2;
% true model parameters
sig(:,:,1) = [1 0; 0 1];
sig(:,:,2) = [0 0; 0 1];
model.Sigma = sig; % covariance matrices
% mismatched parameters
sigt(:,:,1) = [1 0; 0 0];
sigt(:,:,2) = [0 0 ; 0 1];
mmodel.Sigma = sigt; % covariance matrices


% simulate error probability
[P{idx}, Pm{idx}] = mismatchPe(model,mmodel,param);
% simulate upper bound
Pmub{idx} =UBmismatch(model, mmodel, param.sigma2);
% simulate divergence bound
delta{idx} = delta2class(model, mmodel, param.sigma2);
Pubkl{idx} = min(Pm{idx}.Pe+delta{idx}, 1);
% theoretical predictions
[P_params pij_params] = analyseAll(model,mmodel);
d{idx} = P_params.d;

% example c
idx = 3;
rot = @(theta)[cos(theta), - sin(theta); sin(theta), cos(theta)]; % rotation matrix
% true model parameters
sig(:,:,1) = (rot(pi/4)*[1;0])*(rot(pi/4)*[1;0])';
sig(:,:,2) = [0 0; 0 1];
model.Sigma = sig; % covariance matrices
% mismatched parameters
sigt(:,:,1) = sig(:,:,1);
sigt(:,:,2) = (rot(5*pi/6)*[1;0])*(rot(5*pi/6)*[1;0])';
mmodel.Sigma = sigt; % covariance matrices

% simulate error probability
[P{idx}, Pm{idx}] = mismatchPe(model,mmodel,param);
% simulate upper bound
Pmub{idx} =UBmismatch(model, mmodel, param.sigma2);
% simulate divergence bound
delta{idx} = delta2class(model, mmodel, param.sigma2);
Pubkl{idx} = min(Pm{idx}.Pe+delta{idx}, 1);
% theoretical predictions
[P_params pij_params] = analyseAll(model,mmodel);
d{idx} = P_params.d;
Pubkl{idx} = min(Pm{idx}.Pe+delta{idx}, 1);

% example d
idx = 4;
rot = @(theta)[cos(theta), - sin(theta); sin(theta), cos(theta)]; % rotation matrix
% true model parameters
sig(:,:,1) = (rot(pi/4)*[1;0])*(rot(pi/4)*[1;0])';
sig(:,:,2) = [0 0; 0 1];
model.Sigma = sig; % covariance matrices
% mismatched parameters
sigt(:,:,1) = sig(:,:,1);
sigt(:,:,2) = (rot(4*pi/6)*[1;0])*(rot(4*pi/6)*[1;0])';
mmodel.Sigma = sigt; % covariance matrices


% simulate error probability
[P{idx}, Pm{idx}] = mismatchPe(model,mmodel,param);
% simulate upper bound
Pmub{idx} =UBmismatch(model, mmodel, param.sigma2);
% simulate divergence bound
delta{idx} = delta2class(model, mmodel, param.sigma2);
Pubkl{idx} = min(Pm{idx}.Pe+delta{idx}, 1);
% theoretical predictions
[P_params pij_params] = analyseAll(model,mmodel);
d{idx} = P_params.d;
Pubkl{idx} = min(Pm{idx}.Pe+delta{idx}, 1);

%%
figure(1)
NSR = mag2db(1./param.sigma2)./2;
lw = 2;
for i = 1:4
    subplot(2,2,i)
    plot(NSR, log10(Pm{i}.Pe), 'k', 'lineWidth',lw);
    hold on;
    plot(NSR, log10(Pubkl{i}),  'y--', 'lineWidth',lw);
    plot(NSR, log10(Pmub{i}.Pe),  'r', 'lineWidth',lw);
    legend('Pe', 'Pe_{kl}', 'Pe_{UB}', 'Location', 'South')
    grid on 
    AX = [-10, 100, -4, 0.5];
    axis(AX)
    title(sprintf('P(e), d: %d', d{i})) 
end

%% Figure 4
clear; %close all; clc
rng(0,'twister') 

N = 6;
% true model parameters
sig(:,:,1) = diag([1 1 1 0 0 0]);
sig(:,:,2) = diag([0 0 0 1 1 1]);
model.P = [0.5, 0.5]; % priors
model.Sigma = sig; % covariance matrices
model.mu = zeros(2,N)'; % means

% mismatched parameters
mmodel.P = [0.5, 0.5]; % priors
mmodel.mu = zeros(2,N)'; % means

% simulation parameters
param.runs = 8; % repeat the experiments
param.testsamples = 1e5;% number of testing samples per experiment
param.sigma2 = logspace(1,-10,12); % nose variance
param.sigma2factor = 1; % factor of over/under estimation of noise variance

% example 1

% mismatched model
sigt(:,:,1) = diag([1 0 0 0 0 0]);
sigt(:,:,2) = diag([0 0 0 0 0 1]);
mmodel.Sigma = sigt; % covariance matrices

% true error probability analysis
[P, Pm{1}] = mismatchPe(model,mmodel,param);
% theoretical analysis
[P_params pij_params] = analyseAll(model,mmodel);
d{1} = P_params.d;

% example 2

% mismatched model
sigt(:,:,1) = diag([1 1 0 0 0 0]);
sigt(:,:,2) = diag([0 0 0 0 1 1]);
mmodel.Sigma = sigt; % covariance matrices

% true error probability analysis
[P, Pm{2}] = mismatchPe(model,mmodel,param);
% theoretical analysis
[P_params pij_params] = analyseAll(model,mmodel);
d{2} =P_params.d;

% example 3

% mismatched model
sigt(:,:,1) = diag([1 1 1 0 0 0]);
sigt(:,:,2) = diag([0 0 0 1 1 1]);
mmodel.Sigma = sigt; % covariance matrices

% true error probability analysis
[P, Pm{3}] = mismatchPe(model,mmodel,param);
% theoretical analysis
[P_params pij_params] = analyseAll(model,mmodel);
d{3} = P_params.d;

figure(2);
NSR = mag2db(1./param.sigma2)./2;
lw = 2;
plot(NSR, log10(Pm{1}.Pe), 'k', 'lineWidth',lw);
hold on;
plot(NSR, log10(Pm{2}.Pe), 'b', 'lineWidth',lw);
plot(NSR, log10(Pm{3}.Pe), 'r', 'lineWidth',lw);
grid on 
AX = [-10, 80, -4, 0];
axis(AX)
title(sprintf('P(e), d1 = %d, d2 = %d, d3 = %d', d{1},d{2},d{3})) 


