function f1Score = f1Measure(Precision,Recall)
% Measure F1-Score given Precision and Recall of the labels and predicted
% output
% Group E
% Last Modified 27/11/2018 10:34PM

if (size(Precision) ~= size(Recall))
    error('Precision and Recall must be of same size');
end

for ind=1:size(Precision, 1)
    f1Score(ind) = (2 * Precision(ind) * Recall(ind)) / (Precision(ind) + Recall(ind));
end
end