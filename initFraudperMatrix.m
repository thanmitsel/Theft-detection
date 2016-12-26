## Copyright (C) 2016 notroot
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
## @deftypefn {Function File} {@var{retval} =} initializeFraud (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: notroot <notroot@notroot-novo>
## Created: 2016-12-23

function [dstart, fraudDays] = initFraudperMatrix(data)
%Works on a 2D matrix

rate=rand(1);
nRows=size(data,1);
nSample=floor(rate*size(data,1));
rndIDX=randperm(nRows);
newSample=rndIDX(1:nSample);
fraudDays=sort(newSample);

dstart=fraudDays(1);

endfunction
