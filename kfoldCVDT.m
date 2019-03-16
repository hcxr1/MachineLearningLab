function [accuracy,f1Score,Precision,Recall] = kfoldCVDT(kF,examples,targets)
% Perform k-fold CrossValidation
% Input parameter:
% - kF: number of folds
% - discretized features points
% - targets observed
% Outputs: f1-Score, Precision, Recall
% Group E
% Last Modified 27/11/2018 11:03PM

indices = crossvalind('Kfold',targets,kF,'Classes',targets);

for ind = 1:kF
    testdata = (indices == ind)';
    traindata = ~testdata';
    trainSamples = examples(traindata,:);
    trainTargets = targets(traindata);
    Tree = DecisionTreeLearning(trainSamples,1:size(examples,2),trainTargets);
    
    testSamples = examples(testdata,:);
    for jnd = 1 :size(testSamples,1)
        YPredicted(jnd) = classifyTree(Tree,testSamples(jnd,:));
    end
    
    C = confusionmat(targets(testdata),YPredicted');
    
    figure(), confusionchart(C);
    p = (sum(sum(C))-trace(C))/sum(sum(C)); % p = incorrect classification
    correctP(ind) = 100*(1-p);
    incorrectP(ind) = 100*p;
    
    % Precision
    Recall0 = C(1,1)/(C(1,1)+C(1,2));
    Recall1 = C(2,2)/(C(2,1)+C(2,2));
    Recall(:,ind) = [Recall0;Recall1];
    % Recall
    Precision0 = C(1,1)/(C(1,1) + C(2,1));
    Precision1 = C(2,2)/(C(2,2) + C(1,2));
    Precision(:,ind) = [Precision0;Precision1];
    
    % f-Score
    temp = f1Measure(Precision(:,ind),Recall(:,ind));
    f1Score(:,ind) = temp';
end
correctP
incorrectP
accuracy = mean(correctP)
end