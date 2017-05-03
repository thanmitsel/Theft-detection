function [X_norm, mu, sigma] = normalizeMinus_Plus(X)
%NORMALIZEMINUS_PLUS Normalizes the features in X 
%   normalizeMinus_Plus(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

%   Discard a features that have only zeros
if sum(mean(X)==0)~=0
    idx=not(mean(X)==0);
    X=X(:,idx);
end
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);


% ============================================================

end
