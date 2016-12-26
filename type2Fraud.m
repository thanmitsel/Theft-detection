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
## @deftypefn {Function File} {@var{retval} =} type2Fraud (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: notroot <notroot@notroot-novo>
## Created: 2016-12-23

function [f_data,y, F_data,Y] = type2Fraud (data)
% Fraud can be interrapted on days and hours
% with different intensity on days
F_data=data;
Y=zeros(size(data,1),1);
[dstart, fraudDays] = initFraudperMatrix(data);
for i=1:length(fraudDays)
  [tstart, duration, intensity] = initFraudperRow (data);
  F_data(fraudDays(i),tstart:(tstart+duration))=...
    intensity*data(fraudDays(i),tstart:(tstart+duration));
  Y(fraudDays(i),1)=1;
end
f_data_temp=F_data'(:);
f_data=f_data_temp';
y=1;
endfunction
