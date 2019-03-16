function [accuracy,bestParam,bestEpsilon,bestBox,RMSErr] = kfoldCVSVMr(mode,kF,datapoints,labels,boxC,paramList,epsilon)
% Perform kF fold cross validation
% Inputs: 
% mode 
% 1: regression (rbf)
% 2: regression(polynomial)
% 3: regression(linear) --> set epsilon and paramList to 1
% kF number of fold --> [outerfold innerfold]
% 1. Dataset 
% 2. labels 
% 3. BoxConstraint value (slack variable)
% 4. List of possible value for parameter
% 5. Epsilon value
% 6. early Stopping parameter
% 
% Outputs:
% - accuracy
% - bestParameter
% - bestEpsilon
% - best Slack variable value
% - RMSE

% Last Modified 12/1/2018
% Group E

indicesOuter = crossvalind('Kfold',labels,kF(1),'Classes',labels);
bestParam = zeros(kF(1),1);
bestEpsilon = zeros(kF(1),1);
bestBox = zeros(kF(1),1);
% Outer fold
for ind = 1:kF(1)
    testdataO = (indicesOuter == ind)';
    traindataO = ~testdataO';
    prevRate = inf;
    indicesInner = crossvalind('Kfold',labels(traindataO),kF(2),'Classes',labels(traindataO));

    % for all possible parameter values
    for param = paramList
        for ep = epsilon
            for box = boxC
                % Inner Cross-Validation for parameter tuning
                for jnd = 1:kF(2)
                    testdataI = (indicesInner == jnd)';
                    traindataI = ~testdataI';
                    if (mode == 1)
                        Mdl = fitrsvm(datapoints(traindataI,:),labels(traindataI,:),'KernelFunction','rbf','KernelScale',param,'Epsilon',ep,'BoxConstraint',box);
                    elseif (mode == 2)
                        Mdl = fitrsvm(datapoints(traindataI,:),labels(traindataI,:),'KernelFunction','polynomial','PolynomialOrder',param,'Epsilon',ep,'BoxConstraint',box);
                    elseif (mode == 3)
                        Mdl = fitrsvm(datapoints(traindataI,:),labels(traindataI,:),'KernelFunction','linear','BoxConstraint',box);
                    else
                        error('There is no such mode');
                    end
                    
                    YpredI = predict(Mdl,datapoints(testdataI,:));
                    cR(jnd) = sqrt(immse(YpredI,labels(testdataI)));
                end
                classRate = mean(cR) % for regression = rmse, for classification = misclassification rate
                if (classRate < prevRate)
                    bestParam(ind) = param;
                    bestEpsilon(ind) = ep;
                    bestBox(ind) = box;
                    prevRate = classRate;
                end
            end
        end
    end
    
    if (mode == 1)
        Mdl = fitrsvm(datapoints(traindataI,:),labels(traindataI,:),'KernelFunction','rbf','KernelScale',bestParam(ind),'Epsilon',bestEpsilon(ind),'BoxConstraint',bestBox(ind));
    elseif (mode == 2)
        Mdl = fitrsvm(datapoints(traindataI,:),labels(traindataI,:),'KernelFunction','polynomial','PolynomialOrder',bestParam(ind),'Epsilon',bestEpsilon(ind),'BoxConstraint',bestBox(ind));
    elseif (mode == 3)
        Mdl = fitrsvm(datapoints(traindataI,:),labels(traindataI,:),'KernelFunction','linear','BoxConstraint',bestBox(ind));
    else
        error('There is no such mode');
    end
    
    YpredO = predict(Mdl,datapoints(testdataO,:));
    RMSErr(ind) = sqrt(immse(YpredO,labels(testdataO)));
end
accuracy = mean(RMSErr)
end