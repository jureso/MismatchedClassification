% Jure Sokolic, "Mismatch in the Classification of Linear Subspaces:
% Sufficient Conditions for Reliable Classification"
% 
% Copyright @ Jure Sokolic, 2015
% jure.sokolic.13@ucl.ac.uk
clear; close all; clc
rng(0,'twister') 
addpath utils
%% Simulation:
load 'data/Hopkins1RT2RCR.mat' % hopkins data

r= 4; % dimension of subspaces
cov_regularization = 1e-5; % numerical regulrization for the covariance matrices
nruns = 1000; % number of experiment runs
trainsamples = 90; % maximum number of samples used for training

n1 = r:2:trainsamples;  
n2 = r:2:trainsamples; 
n3 = [4,30,70,90];

% run the simulation
%[results] = hopkinsSimulation(X, r, trainsamples, nruns,cov_regularization,n1,n2,n3)
%save('data/results', 'results')
%% Results
load('data/results')

% index for n_3, i.e. n_3 = n3(i3);
i3 = 3;

% collect all runs 
errtemp = zeros(length(n1),length(n2), nruns);
boundtemp = zeros(length(n1),length(n2), nruns);
for i = 1:nruns
    errtemp(:,:,i) = results.errMismatch{i}(:,:,i3)> 0; % true error probability phase transitions
    boundtemp(:,:,i) = ~results.boundPT{i}(:,:,i3); % bound phase transition
end
errtemp = sort(errtemp,3,'ascend');
boundtemp = sort(boundtemp,3,'ascend');

% probabilities and appropriate indices
pp_1= 0.8;
pp_2= 0.9;
idx1 =round(nruns*pp_1);
idx2 =round(nruns*pp_2);

% display phase transition
figure; 
subplot(1,2,1)
imagesc(flipud(errtemp(:,:,idx1)), [0,1])
hold on;
c1 = contour(flipud(boundtemp(:,:,idx1)),1, 'r');
title(sprintf('Phase transitions: p_p = %.1f, n_3 = %d', pp_1, n3(i3)))
subplot(1,2,2)
imagesc(flipud(errtemp(:,:,idx2)), [0,1])
hold on;
c2= contour(flipud(boundtemp(:,:,idx2)),1, 'r');
title(sprintf('Phase transitions: p_p = %.1f, n_3 = %d', pp_2, n3(i3)))

