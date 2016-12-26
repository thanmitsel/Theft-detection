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
## @deftypefn {Function File} {@var{retval} =} type4Fraud (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: notroot <notroot@notroot-novo>
## Created: 2016-12-25

function [factor] = type4Fraud (data)
% factor is the average of the intensity per fraud duration on day
[dstart, fraudDays] = initFraudperMatrix(data);
for i=1:length(fraudDays)
  factor(i,1)=0;
  [tstart, duration, intensity] = initFraudperRow(data); %intensity not needed
  if tstart~=24
    for j=tstart:(tstart+duration)
      [patates, agouria, hour_intensity] = initFraudperRow(data); %patates-agouria not needed (expensive)
        factor(i,1)=factor(i,1)+hour_intensity;    
     end
  else % when start is 24th hour
    duration=1; 
    factor(i,1)=intensity;
  end
  factor(i,1)=factor(i,1)/duration;
end
endfunction
