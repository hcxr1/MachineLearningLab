% Assignment 3: SVM
% Group E
% Last Modified: 1/12/2018 - 5:23PM
% dataset: regression data
clc;
clear all;
close all;

addpath 'regression headpose';
load 'facialPoints.mat'
load 'headpose.mat'
labels = pose(:,6)';
S = size(points);

datapoints = reshape(points,S(1)*S(2),S(3));
datapoints = datapoints';
labels = labels';
mode = 2;
switch (mode)
    case 1
        [accuracy,bestParam,bestBox,correctP]=kfoldCVSVMr(1,[5 2],datapoints,labels,[0.01 0.1 1 10 100 1000], [0.01 0.1 1 10 100 1000], [0.01 0.1 1 10 100 1000]);
        Mdl = fitrsvm(datapoints,labels,'KernelFunction','rbf','KernelScale',mean(bestParam),'Epsilon',mean(bestEpsilon),'BoxConstraint',mean(bestBox));
    case 2
        [accuracy,bestParam,bestBox,correctP]=kfoldCVSVMr(2,[5 2],datapoints,labels,[0.01 0.1 1 10 100 1000], [2 3 4 5], [0.01 0.1 1 10 100 1000]);
        Mdl = fitrsvm(datapoints,labels,'KernelFunction','polynomial','PolynomialOrder',floor(mean(bestParam)),'Epsilon',mean(bestEpsilon),'BoxConstraint',mean(bestBox));
    case 3    
        [accuracy,bestParam,bestBox,correctP]=kfoldCVSVMr(3,[5 2],datapoints,labels,[0.01 0.1 1 10 100 1000], 1, [0.01 0.1 1 10 100 1000]);
        Mdl = fitrsvm(datapoints,labels,'KernelFunction','linear','BoxConstraint',mean(bestBox));
    case 4
        Mdl = fitrsvm(datapoints,labels,'OptimizeHyperparameters','auto',...
            'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
            'expected-improvement-plus'));
end
sv = Mdl.SupportVectors;
Ypred = predict(Mdl,datapoints);
rmse = sqrt(immse(Ypred,labels))

switch (mode)
    case 1
        save resultregheadposeSVM1.mat
    case 2
        save resultregheadposeSVM2.mat
    case 3
        save resultregheadposeSVM3.mat
end
