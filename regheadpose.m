% Lab Sessions 3 &4 (Assignment 1): Artificial Neural Networks
% Group E
% Last Modified: 24/10/2018 - 10:07AM
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

% nntool
NET = newff(datapoints,labels,1,{'purelin'},'trainlm');%[5,3],{'tansig' 'purelin'},'trainlm'); % Create a network
NET.trainParam.epochs = 50;
NET.trainParam.lr = 1.01;
NET.trainParam.mu = 0.00001;
NET.trainParam.show = 25;
NET.trainParam.goal = 0;
NET.trainParam.max_fail = 6;
[NET,TR] = train(NET,datapoints,labels); % Training process

Pred_T = sim(NET,datapoints); % Predicted Target

% outputs = NET(datapoints);
errors = Pred_T - labels;
perf = perform(NET,Pred_T,labels)
rmse = sqrt(immse(Pred_T,labels))

figure(),
plotperform(TR);

figure(),
plot(labels,'r--')
hold on
plot(Pred_T,'bx');
hold off
xlabel('Samples');
ylabel('Points');
title('Y Predicted vs Y Observed');
legend('Y observed','Y Predicted');

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% 10 fold cross validation
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
[a,c,p] = kfoldCVNN(2,10,datapoints,labels,1,{'purelin'},[50,1.01,10,0.1,0.000001]);%[5,3],{'tansig' 'purelin'},'trainlm',[50,1.01,10,0.1,0.000001]);

save resultregheadposeANN.mat
