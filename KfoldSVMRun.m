## Copyright (C) 2017 notroot
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} KfoldSVMRun (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: notroot <notroot@notroot-novo>
## Created: 2017-01-23

function [prec, rec, acc, F1] = KfoldSVMRun (features,Y, K, C, gamma)
% Get some initializations
sum_prec=0;
sum_rec=0;
sum_acc=0;
sum_F1=0;

% Shuffle feature rows
idx=randperm(size(features, 1));
features=features(idx, :);
Y=Y(idx);

arguments=['-t ' num2str(2) ' -g ' num2str(gamma) ' -c ' num2str(C)]; 
c = cvpartition(size(features,1), 'KFold', K); 
for i=1:K
  idx_train=~test(c, K);
  idx_val=test(c, K);
  
  Xtrain=features(idx_train,:);
  Ytrain=Y(idx_train);
  Xval=features(idx_val,:);
  Yval=Y(idx_val);
  
  model=svmtrain(Ytrain,Xtrain,arguments);
  predictions= svmpredict(Yval,Xval,model);
  
  [prec,rec,acc,F1] = confusionMatrix (Yval, predictions);
  sum_prec=sum_prec+prec;
  sum_rec=sum_rec+rec;
  sum_acc=sum_acc+acc;
  sum_F1=sum_F1+F1;
end
prec=sum_prec/K;
rec=sum_rec/K;
acc=sum_acc/K;
F1=sum_F1/K;
endfunction
