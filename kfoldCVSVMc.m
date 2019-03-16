function [accuracy,bestParam,bestBox,correctP] = kfoldCVSVMc(mode,kF,datapoints,labels,boxC,paramList)
% Perform kF fold cross validation
% Inputs: 
% mode 
% 1: classification (rbf) 
% 2: classification (polynomial)
% 3: classification (linear) --> set epsilon and paramList to 1
% kF number of fold --> [outerfold innerfold]
% 1. Dataset 
% 2. labels 
% 3. BoxConstraint value (slack variable)
% 4. List of possible value for parameter
% 
% Outputs:
% - Accuracy
% - BestParameter
% - Best Slack value
% - Classification rate


% Last Modified 12/1/2018
% Group E

indicesOuter = crossvalind('Kfold',labels,kF(1),'Classes',labels);
bestParam = zeros(kF(1),1);
bestBox = zeros(kF(1),1);

% Outer fold
for ind = 1:kF(1)
    testdataO = (indicesOuter == ind)';
    traindataO = ~testdataO';
    prevRate = inf;
    indicesInner = crossvalind('Kfold',labels(traindataO),kF(2),'Classes',labels(traindataO));
    % for all possible parameter values
    for param = paramList
        for box = boxC
            % Inner Cross-Validation for parameter tuning
            for jnd = 1:kF(2)
                testdataI = (indicesInner == jnd)';
                traindataI = ~testdataI';
                
                if (mode == 1)
                    Mdl = fitcsvm(datapoints(traindataI,:),labels(traindataI),'KernelFunction','rbf','KernelScale',param,'BoxConstraint',box);
                elseif (mode == 2)
                    Mdl = fitcsvm(datapoints(traindataI,:),labels(traindataI),'KernelFunction','polynomial','PolynomialOrder',param,'BoxConstraint',box);
                elseif (mode == 3)
                    Mdl = fitcsvm(datapoints(traindataI,:),labels(traindataI),'KernelFunction','linear','BoxConstraint',box);
                else
                    error('There is no such mode');
                end
                
                YpredI = predict(Mdl,datapoints(testdataI,:));
                CM = confusionmat(labels(testdataI),YpredI);
                %                 figure(), confusionchart(CM);
                cR(jnd) = (sum(sum(CM))-trace(CM))/sum(sum(CM));
            end
            classRate = mean(cR) % for regression = rmse, for classification = misclassification rate
            if (classRate < prevRate)
                bestParam(ind) = param;
                prevRate = classRate;
                bestBox(ind) = box;
            end
        end
    end
    
    if (mode == 1)
        Mdl = fitcsvm(datapoints(traindataO,:),labels(traindataO),'KernelFunction','rbf','KernelScale',bestParam(ind),'BoxConstraint',bestBox(ind));
    elseif (mode == 2)
        Mdl = fitcsvm(datapoints(traindataO,:),labels(traindataO),'KernelFunction','polynomial','PolynomialOrder',bestParam(ind),'BoxConstraint',bestBox(ind));
    elseif (mode == 3)
        Mdl = fitcsvm(datapoints(traindataI,:),labels(traindataI),'KernelFunction','linear','BoxConstraint',bestBox(ind));
    else
        error('There is no such mode');
    end
    
    YpredO = predict(Mdl,datapoints(testdataO,:));
    C = confusionmat(labels(testdataO),YpredO);
    figure(), confusionchart(C);
    p = trace(C)/sum(sum(C));
    incorrectP(ind) = (1-p)*100;
    correctP(ind) = p*100;
end
accuracy = mean(correctP) % for classification: correct Percentage; for Regression: Root Means Square Error
end