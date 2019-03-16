% Lab Sessions 3 &4 (Assignment 1): Artificial Neural Networks
% Group E
% Last Modified: 11/08/2018 - 02:34AM
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

% nntool
NET = newff(datapoints,labels,1,{'tansig'},'trainlm');%[33,17,9,5,3,2],{'logsig' 'tansig' 'tansig' 'tansig' 'tansig' 'tansig' 'tansig'},'trainlm'); % Create a network
NET.trainParam.epochs = 100;
NET.trainParam.lr = 1.01;
NET.trainParam.mu = 0.00001;
NET.trainParam.max_fail = 6;
NET.trainParam.show = 25;
NET.trainParam.goal = 0;

[NET,TR] = train(NET,datapoints,labels); % Training process

Pred_T = sim(NET,datapoints); % Predicted Target
indx = find(Pred_T >= 0.5);
jndx = find(Pred_T < 0.5);
outputs(indx) = 1;
outputs(jndx) = 0;

% outputs = NET(datapoints);
errors = Pred_T - labels;
perf = perform(NET,Pred_T,labels)

figure(),
plotperform(TR);

figure(),
plot(labels,'r--')
hold on
plot(Pred_T,'bx');
hold off
xlabel('Samples');
ylabel('Classification');
title('Y Predicted vs Y Observed');
legend('Y observed','Y Predicted');

cm = confusionmat(labels',outputs');
c = (sum(sum(cm))-trace(cm))/sum(sum(cm));
incorrectPercentage = c*100
correctPercentage = 100*(1-c)

figure(), C = confusionchart(cm);
% figure(), plotregression(labels',outputs');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% 10 fold Cross-Validation
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
kF = 10;
[a,c,p,cmat] = kfoldCVNN(1,kF,datapoints,labels,1,{'tansig'},'trainlm',[100,1.01,1.05,0.7,0.00001]);

save resultbinsmileANN.mat