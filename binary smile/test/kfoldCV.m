function [accuracy,correctP,incorrectP,perf,Outputs,errors] = kfoldCV(varargin);
% Perform 10 fold cross validation
% Inputs: 
% 1. Dataset 
% 2. label 
% 3..end: network parameter (if left empty then a default value for each parameters will be used)
% Outputs:  Average Accuracy, Percentage of Correct Classification and Percentage of incorrect Classificationperformance, predicted value, error,
% training parameter (inputted as an array):
% 1. epochs
% 2. learning rate
% 3. learning rate increase
% 4. learning rate decrease
% 5. mu
% 6. maximum validation fail
% 7. show
% 8. goal
% Last Modified 7/11/2018
% Group E
datapoints= varargin{1};
labels = varargin{2};
trainP = varargin{end};
indices = crossvalind('Kfold',labels,10,'Classes',labels);

for i = 1:10
    testdata = (indices == i)';
    traindata = ~testdata';
    
    NET = newff(datapoints(:,traindata),labels(:,traindata),varargin{3:end-1}); % Create a network
    NET.trainParam.epochs = trainP(1);
    NET.trainParam.lr = trainP(2);
    NET.trainParam.lr_inc = trainP(3);
    NET.trainParam.lr_dec = trainP(4);
    NET.trainParam.mu = trainP(5);
    NET.trainParam.max_fail = trainP(6);
    NET.trainParam.show = trainP(7);
    NET.trainParam.goal = trainP(8);
    NET.divideParam.trainRatio=0.8;
    NET.divideParam.valRatio=0.1;
    NET.divideParam.testRatio=0.1;
    
    [NET,TR] = train(NET,datapoints(:,traindata),labels(traindata)); % Training process
    Pred_T(i,:) = sim(NET,datapoints(:,testdata));
    
    indx = find(Pred_T(i,:) >= 0.5);
    jndx = find(Pred_T(i,:) < 0.5);
    Outputs(i,indx) = 1;
    Outputs(i,jndx) = 0;
    errors(i,:) = Outputs(i,:) - labels(testdata);
    perf(i) = perform(NET,Outputs(i,:),labels(testdata));
    
    cm = confusionmat(labels(testdata)',Outputs(i,:)')
    c = (sum(sum(cm))-trace(cm))/sum(sum(cm));
    correctP(i) = 100*(1-c);
    incorrectP(i) = 100*c;
end
accuracy = mean(correctP)
end