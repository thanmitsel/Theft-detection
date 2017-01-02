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
## @deftypefn {Function File} {@var{retval} =} crossValind (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: notroot <notroot@notroot-novo>
## Created: 2017-01-01

function [trainidx, testidx] = crossValind (initial_size, test_rate)
% Outputs logical vectors with 1 when used in training or testing 
trainidx=zeros(initial_size,1);
testidx=zeros(initial_size,1);

test_size=round(test_rate*initial_size);
test=randperm(initial_size,test_size);
testidx(test)=1;

testidx=logical(testidx);
trainidx=~testidx;
endfunction
