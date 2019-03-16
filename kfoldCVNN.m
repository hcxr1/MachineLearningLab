function [accuracy,correctP,perf,cmat] = kfoldCVNN(mode,kF,varargin);
% Perform kF fold cross validation (Classification)
% Inputs: 
% mode 1: classification 2: regression
% kF number of fold
% 1. Dataset 
% 2. labels 
% 3..end: network parameter (if left empty then a default value for each parameters will be used)
% Outputs:  Average Accuracy, Percentage of Correct Classification, performance
% training parameter (inputted as an array):
% 1. epochs
% 2. learning rate
% 3. learning rate increase
% 4. learning rate decrease
% 5. mu
% Last Modified 8/11/2018
% Group E
if (nargin < 6)
    error('Not enough arguments!');
end
datapoints= varargin{1};
temp = varargin{2};
if (mode == 1)
    if (max(temp) == 1)
        labels = varargin{2};
        y_observed = labels;
    elseif (max(temp) >= 1)
        %     perform i of k for multiclass
        labels = zeros(size(temp,2),max(temp));
        for i = 1:size(temp,2)
            indx = temp(i);
            labels(i,indx) = 1;
        end
        y_observed = temp;
        labels = labels';
    end
    sz = max(temp);
elseif (mode == 2)
    labels = temp;
    y_observed = labels;
end
    
trainP = varargin{end};
indices = crossvalind('Kfold',y_observed,kF,'Classes',y_observed);

for i = 1:kF
    testdata = (indices == i)';
    traindata = ~testdata';
    
    NET = newff(datapoints(:,traindata),labels(:,traindata),varargin{3:end-1}); % Create a network
    NET.trainParam.epochs = trainP(1);
    NET.trainParam.lr = trainP(2);
    NET.trainParam.mu_inc = trainP(3);
    NET.trainParam.mu_dec = trainP(4);
    NET.trainParam.mu = trainP(5);
%     NET.trainParam.max_fail = trainP(6);  
    NET.divideFcn = 'dividetrain';
    
    [NET,TR] = train(NET,datapoints(:,traindata),labels(:,traindata)); % Training process
    Pred_T = sim(NET,datapoints(:,testdata));
    if (mode == 1)
        if (sz == 1)
            indx = find(Pred_T >= 0.5);
            jndx = find(Pred_T < 0.5);
            y_pred(indx) = 1;
            y_pred(jndx) = 0;
        else
            outputs = zeros(size(Pred_T));
            y_pred = zeros(size(y_observed(testdata)));
            for j = 1:size(Pred_T,2)
                [m,ind] = max(Pred_T(:,j));
                outputs(ind,j) = 1;
                for k = 1 : sz
                    if (outputs(k,j) == 1)
                        y_pred(j) = k;
                    end
                end
            end
        end
    elseif (mode == 2)
        y_pred = Pred_T;
    end
    errors = y_pred - y_observed(testdata);
    perf(i) = perform(NET,y_pred,y_observed(testdata));
    if (mode == 1)
        cm = confusionmat(y_observed(testdata)',y_pred') % confusion matrix
        cmat(i,:,:) =cm;
        c = (sum(sum(cm))-trace(cm))/sum(sum(cm)); % c = incorrect classification
        correctP(i) = 100*(1-c);
        incorrectP(i) = 100*c;
        figure(), confusionchart(cm);
    elseif (mode == 2)
        correctP(i) = sqrt(immse(y_pred,y_observed(testdata))); % rmse
    end
end
accuracy = mean(correctP) % for classification: correct Percentage; for Regression: Root Means Square Error
end