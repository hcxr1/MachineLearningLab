% Assignment 2: Decision Tree
% Group E
% Last Modified: 28/11/2018 12:55PM
% Dataset: Binary Smile

clc;
clear all;
close all;

addpath 'binary smile';
load 'facialPoints.mat';
load 'labels.mat';

dp = reshape(points,132,150);

% Discretized continuous input data
for jnd = 1: size(dp,1)
    threshold(jnd) = median(dp(jnd,:));
    for ind = 1:size(dp,2)
        if (dp(jnd,ind) < threshold(jnd))
            discPoints(jnd,ind) = 0;
        else
            discPoints(jnd,ind) = 1;
        end
    end
end
discPoints = discPoints';

tree = DecisionTreeLearning(discPoints, 1:132, labels);
DrawDecisionTree(tree);

for ind = 1: size(dp,2)
    YPred(ind) = classifyTree(tree,discPoints(ind,:));
end
figure(), plot(YPred,'*');
hold on
plot(labels);
hold off
legend('predicted','target');
title('Target Label vs Predicted Result');
xlabel('sample');

% Confusion Matrix
C = confusionmat(labels,YPred');
figure(), confusionchart(C);
p = (sum(sum(C))-trace(C))/sum(sum(C)); % c = incorrect classification
correctP = 100*(1-p)
incorrectP = 100*p

% Recall
Recall0 = C(1,1)/(C(1,1)+C(1,2));
Recall1 = C(2,2)/(C(2,1)+C(2,2));
Recall = [Recall0;Recall1]

% Precision
Precision0 = C(1,1)/(C(1,1) + C(2,1)); 
Precision1 = C(2,2)/(C(2,2) + C(1,2));
Precision = [Precision0;Precision1]

% f1-Score
f1Score = f1Measure(Precision,Recall)

% Perform Cross-Validation
[accuracy,f1_Score,kPrecision,kRecall] = kfoldCVDT(10,discPoints,labels)

save tree.mat tree

save resultbinsmileTree.mat