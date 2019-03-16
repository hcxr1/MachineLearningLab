function tree = DecisionTreeLearning( features, attributes, labels )
%DecisionTreeLearning generates a decision tree based on ID3 algorithm
% given the following input:
% - features: discretized input features
% - attributes
% - targeted labels

% Group E
% Last Modified 27/11/2018 10:34PM

if isempty(attributes)
    % if there are no more attributes to split the tree on return the
    % majority value of the labels
    tree.kids = cell(0);
    tree.class = mode(labels);
else
    bestAttribute = chooseAttribute(features, attributes, labels);
    k = find(attributes == bestAttribute);
    kids = cell(0);
    
    for v=0:1
        tempExamples = [];
        tempTargets = [];
        
        for i=1:size(features, 1)
            if features(i, k) == v             
               % Build up new examples and binaryTargets
               tempExamples = [tempExamples; features(i, :)];
               tempTargets = [tempTargets; labels(i)];
            end
        end
        
        if isempty(tempExamples)
            tree.kids = cell(0);
            tree.class = mode(labels);
            return
        else
            kid = DecisionTreeLearning(tempExamples, attributes(find(attributes ~= bestAttribute)), tempTargets);            
            kids{size(kids, 1) + 1} = kid;
        end
    end
    
    % New tree with bestAttribute as its root
    tree.op = bestAttribute;
    tree.kids = kids;
end

function bestAttribute  = chooseAttribute(examples, attributes, targets)
%ChooseAttribute measures how good each attribute in the set is based on
%the highest gain
% Input Parameter:
% - discretized examples
% - attributes
% - targets
% Last Modified 27/11/2018
% Group E

bestAttribute = attributes(1);
maxGain = Inf;

for a=attributes
    p0 = 0;
    p1 = 0;
    n0 = 0;
    n1 = 0;
    
    % Calculate the number of positive and negative examples
    for i=1:size(examples, 1)
        if examples(i, a) == 0
            if targets(i) == 1
                p0 = p0 + 1;
            else
                n0 = n0 + 1;
            end
        else
            if targets(i) == 1
                p1 = p1 + 1;
            else
                n1 = n1 + 1;
            end
     end
    end
    
    I = @(p,n) - (p / p + n) * log2(p / p + n) - (n / p + n) * log2(n / p + n);
    Remainder = @( p0, p1, n0, n1 )(p0 + n0) / ( p0 + p1 + n0 + n1) * I(p0, n0) + (p1 + n1) / (p0 + p1 + n0 + n1) * I(p1, n1);
    gain = I(p0, n0)+ I(p1,n1) - Remainder(p0, p1, n0, n1);

    if gain < maxGain % since the gain is negative, to find the max means to find the most negative
        bestAttribute = a;
        maxGain = gain;
    end
end