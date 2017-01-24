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
## @deftypefn {Function File} {@var{retval} =} frauDetails (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: notroot <notroot@notroot-novo>
## Created: 2017-01-23

function [kWh_count, time_count, kWh_rate, time_rate] = frauDetails (Data, F_data)
kWh_count=sum(sum(Data-F_data));
time_count=sum(sum(Data~=F_data));
time_rate=time_count/(size(Data,1)*size(Data,2));
kWh_rate=kWh_count/sum(sum(Data)); 
endfunction
