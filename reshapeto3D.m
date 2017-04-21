funtion [X]=reshapreto3D(X_full, nlay)
% Reshape from 2D to 3D
% nlay is the size of the 3rd dimension
[row,col] = size(X_full);
X   = permute(reshape(Xn_full',[col,row/nlay,nlay]),[2,1,3]);
