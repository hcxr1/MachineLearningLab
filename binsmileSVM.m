% Assignment 3: SVM
% Group E
% Last Modified: 1/12/2018 - 5:23PM
% dataset: binary smile
clc;
clear all;
close all;

addpath 'binary smile';
load 'facialPoints.mat'
load 'labels.mat'

labels = labels'; % every column = 1 sample (m)
S = size(points);
datapoints = reshape(points,S(1)*S(2),S(3));
datapoints = datapoints';
labels = labels';
mode = 3;

switch (mode)
    case 1
        [accuracy,bestParam,bestBox,correctP]=kfoldCVSVMc(1,[5 2],datapoints,labels,[0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000], [0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000]);
        Mdl = fitcsvm(datapoints,labels,'KernelFunction','rbf','KernelScale',mean(bestParam),'BoxConstraint',mean(bestBox));
    case 2
        [accuracy,bestParam,bestBox,correctP]=kfoldCVSVMc(2,[5 2],datapoints,labels,[0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000], [2 3 4 5]);
        Mdl = fitcsvm(datapoints,labels,'KernelFunction','polynomial','PolynomialOrder',floor(mean(bestParam)),'BoxConstraint',mean(bestBox));
    case 3
        [accuracy,bestParam,bestBox,correctP]=kfoldCVSVMc(3,[5 2],datapoints,labels,[0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000], 1);
        Mdl = fitcsvm(datapoints,labels,'KernelFunction','linear','BoxConstraint',mean(bestBox));
    case 4
        Mdl = fitcsvm(datapoints,labels,'OptimizeHyperparameters','auto',...
            'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
            'expected-improvement-plus'));
end

sv = Mdl.SupportVectors;
Ypred = predict(Mdl,datapoints);
C = confusionmat(labels,Ypred);
figure(), confusionchart(C);
p = trace(C)/sum(sum(C));
classificationRate = p*100
misclassificationRate = (1-p)*100

switch (mode)
    case 1
        save resultbinsmileSVM1.mat
    case 2
        save resultbinsmileSVM2.mat
    case 3
        save resultbinsmileSVM3.mat
end