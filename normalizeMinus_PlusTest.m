function [X_norm] = normalizeMinus_PlusTest(X, mu, sigma)

if sum(mean(X)==0)~=0
    idx=not(mean(X)==0);
    X=X(:,idx);
end

X_norm = bsxfun(@minus, X, mu);


X_norm = bsxfun(@rdivide, X_norm, sigma);