function labelPredicted = classifyTree(tree, example)
% This function takes tree structure and test example as input and gives
% output of the predicted label (1 or 0)
% Group E
% Last Modified 27/11/2018 10:34PM

if isempty(tree.kids)
    labelPredicted = tree.class;
else
    if example(tree.op) == 0
        kid = tree.kids{1};
        labelPredicted = classifyTree(kid, example);
    else
        kid = tree.kids{2};
        labelPredicted = classifyTree(kid, example);
    end
end