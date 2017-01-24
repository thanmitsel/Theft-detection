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
## @deftypefn {Function File} {@var{retval} =} findSVMparams (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: notroot <notroot@notroot-novo>
## Created: 2017-01-20

function [C, gamma] = findSVMparams (Xtrain, Ytrain, Xval, Yval)

C_test=[0.01 0.03 0.1 0.3 1 3 10 30];
gamma_test=[0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30];

for i=1:length(C_test)
  for j=1:length(gamma_test)
        arguments=['-t ' num2str(2) ' -g ' num2str(gamma_test(j)) ' -c ' num2str(C_test(i))]; 
        model=svmtrain(Ytrain,Xtrain,arguments);
        predictions= svmpredict(Yval,Xval,model);
        error= mean(double(predictions ~= Yval));
        if i==1 && j==1
            min_error=error;
        end
        if error<min_error
            min_error=error;
            C=C_test(i);
            gamma=gamma_test(j);
        end
     end
 end
endfunction

