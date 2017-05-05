function [trend_coefs]=get_Trend(X_1)
% find trend linear trend
trend_coefs=zeros(size(X_1,3),2);

t=[1:size(X_1,1)]';
% linear trend
X=[ones(size(X_1,1),1) t];
for i=1:size(X_1,3)
    y_av=X_1(:,1,i);
    y_std=X_1(:,2,i);
    b_av=X\y_av; %
    b_std=X\y_std; %
    tH_av=X*b_av;
    tH_std=X*b_std;
    trend_coefs(i,1)=b_av(2,1);
    trend_coefs(i,2)=b_std(2,1);
end

% Plot one Trend if so better plot real resuts too
% h2=plot(tH/1000, 'r', 'LineWidth', 2); 
% legend(h2, 'Linear Trend Estimate');