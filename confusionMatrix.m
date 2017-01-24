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
## @deftypefn {Function File} {@var{retval} =} confusionMatrix (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: notroot <notroot@notroot-novo>
## Created: 2017-01-20

function [prec,rec,acc,F1] = confusionMatrix (yval, pred)
    
tp=sum((pred==yval)&(pred==1));
tn=sum((pred==yval)&(pred==0));
fp=sum((pred==1)&(yval==0));
fn=sum((pred==0)&(yval==1));

prec=tp/(tp+fp);
rec=tp/(tp+fn);
acc=((tp+tn)/(tp+tn+fp+fn));
F1=2*prec*rec/(prec+rec);
    

endfunction
