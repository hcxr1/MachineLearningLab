% Perform TTEST2 on binary smile dataset
load resultbinsmileSVM1.mat
YpredGauss = Ypred;

load resultbinsmileSVM2.mat
YpredPoly = Ypred;

load resultbinsmileSVM3.mat
YpredLin = Ypred;

load resultbinsmileTree.mat
YpredTree = YPred;

load resultbinsmileANN.mat
YpredANN = outputs;

% Comparison 1 Gaussian vs Polynomial
[h1,pval1,ci1,stats1] = ttest2(YpredGauss,YpredPoly);

% Comparison 2 Gaussian vs Linear
[h2,pval2,ci2,stats2] = ttest2(YpredGauss,YpredLin);

% Comparison 3 Polynomial vs Linear
[h3,pval3,ci3,stats3] = ttest2(YpredPoly,YpredLin);

% Comparison 4 best SVM Gaussian vs Tree 
[h4,pval4,ci4,stats4] = ttest2(YpredGauss,YpredTree);

% Comparison 5 best SVM Gaussian vs ANN 
[h5,pval5,ci5,stats5] = ttest2(YpredGauss,YpredANN);

% Comparison 6 best SVM Linear vs Tree 
[h6,pval6,ci6,stats6] = ttest2(YpredLin,YpredTree);

% Comparison 7 best SVM Linear vs ANN 
[h7,pval7,ci7,stats7] = ttest2(YpredLin,YpredANN);

% Comparison 6 best SVM Polynomial vs Tree 
[h8,pval8,ci8,stats8] = ttest2(YpredPoly,YpredTree);

% Comparison 7 best SVM Polynomial vs ANN 
[h9,pval9,ci9,stats9] = ttest2(YpredPoly,YpredANN);