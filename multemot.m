% Lab Sessions 3 &4 (Assignment 1): Artificial Neural Networks
% Group E
% Last Modified: 11/08/2018 - 02:34AM
% dataset: multiclass emotion
clc;
close all;
clear all;

addpath 'multiclass emotion';
load 'emotions_data.mat'

x = x';
y = y';

label = zeros(size(y,2),6);
for i = 1:size(y,2)
   switch(y(i))
       case 1
           label(i,:) = [1 0 0 0 0 0];
       case 2
           label(i,:) = [0 1 0 0 0 0];
       case 3
           label(i,:) = [0 0 1 0 0 0];
       case 4
           label(i,:) = [0 0 0 1 0 0];
       case 5
           label(i,:) = [0 0 0 0 1 0];
       case 6
           label(i,:) = [0 0 0 0 0 1];
   end
end
label = label';
% nntool
NET = newff(x,label,[136 6],{'tansig' 'tansig'},'trainlm');% [34,26,17,9,6],{'logsig' 'tansig' 'tansig' 'tansig' 'tansig'},'trainlm'); % Create a network
NET.trainParam.epochs = 100;
NET.trainParam.lr = 1.01;
NET.trainParam.mu = 0.0001;
NET.trainParam.show = 25;
NET.trainParam.goal = 0;
[NET,TR] = train(NET,x,label); % Training process

Pred_T = sim(NET,x); % Predicted Target

outputs = zeros(size(Pred_T));
for i = 1:size(Pred_T,2)
    [m,ind] = max(Pred_T(:,i));
    outputs(ind,i) = 1;
    if (outputs(1,i) == 1) y_pred(i) = 1;
    elseif (outputs(2,i) == 1) y_pred(i) = 2;
    elseif (outputs(3,i) == 1) y_pred(i) = 3;
    elseif (outputs(4,i) == 1) y_pred(i) = 4;
    elseif (outputs(5,i) == 1) y_pred(i) = 5;
    elseif (outputs(6,i) == 1) y_pred(i) = 6;
    end
end

% y_pred = NET(x);
errors = y_pred - y;
perf = perform(NET,y_pred,y)

figure(),
plotperform(TR);

figure(),
plot(y,'r--')
hold on
plot(y_pred,'bx');
hold off
xlabel('Samples');
ylabel('Classification');
title('Y Predicted vs Y Observed');
legend('Y observed','Y Predicted');

cm = confusionmat(y',y_pred');
c = (sum(sum(cm))-trace(cm))/sum(sum(cm));
incorrectPercentage = c*100
correctPercentage = 100*(1-c)
figure(), C = confusionchart(cm);
% figure(), plotregression(y',y_pred');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% 10 fold cross validation
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
 [a,c,p,cmat] = kfoldCVNN(1,5,x,y,[136 6],{'tansig' 'tansig'},'trainlm',[100,0.01,10,0.1,0.0001]);% [34,26,17,9,6],{'logsig' 'tansig' 'tansig' 'tansig' 'tansig'},[10,0.01,1.05,0.7,0.00001]);
