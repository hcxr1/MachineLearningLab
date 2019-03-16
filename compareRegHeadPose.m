% Perform ttest2 on regression headpose dataset
load resultregheadposeSVM1.mat
YpredGauss = Ypred;

load resultregheadposeSVM2.mat
YpredPoly = Ypred;

load resultregheadposeSVM3.mat
YpredLin = Ypred;

load resultregheadposeANN.mat
YpredANN = Pred_T;

% Comparison 1 Gaussian vs Polynomial
[h1,pval1,ci1,stats1] = ttest2(YpredGauss,YpredPoly);

% Comparison 2 Gaussian vs Linear
[h2,pval2,ci2,stats2] = ttest2(YpredGauss,YpredLin);

% Comparison 3 Polynomial vs Linear
[h3,pval3,ci3,stats3] = ttest2(YpredPoly,YpredLin);

% Comparison 4 best SVM Gaussian vs ANN
[h4,pval4,ci4,stats4] = ttest2(YpredGauss,YpredANN);

% Comparison 5 best SVM Polynomial vs ANN
[h5,pval5,ci5,stats5] = ttest2(YpredPoly,YpredANN);

% Comparison 4 best SVM Linear vs ANN
[h6,pval6,ci6,stats6] = ttest2(YpredLin,YpredANN);