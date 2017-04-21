function [X_full, Y_full]=unrollto2D(X,Y2D)
% Unroll 3D and 2D
X_full= permute(X,[1 3 2]);
X_full= reshape(X_full,[],size(X,2),1);
Y_full=Y2D(:); 
