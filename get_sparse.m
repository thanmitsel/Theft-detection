function [Y_s,X_s]=get_sparse(Y,X)
X_sparse=sparse(X);
libsvmwrite('libSVM.train',Y, X_sparse);
[Y_s, X_s]=libsvmread('libSVM.train');
end
