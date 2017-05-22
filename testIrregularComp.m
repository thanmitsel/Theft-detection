%% Apply Parametric trend estimation to Frauds to check
    z=2000;
    r_cons=randi(size(hh,1),z,1);
    somehh=hh(r_cons,:);
    someID=ID(r_cons,:);

    % Convertions
    [h, H]=convertHours3D(somehh);

    %% Fraud Initialization
    % Create Fraud data
    F_data3D=H;
    Y2D=zeros(size(H,1),size(H,3));
    one_H=zeros(size(H(:,:,1)));

% fraud_rate=0.05; % Percentage of consumers who fraud
fraud_rate=0.1; % Percentage of consumers who fraud
[normal_idx, fraud_idx] = crossvalind('HoldOut', size(H,3), fraud_rate); % Keep in mind crossval floors the rate
thiefs=find(fraud_idx==1);
for i=1:size(thiefs,1)    
    intensity=1-betarnd(6,3); % beta distribution
    dstart=floor(normrnd(size(one_H,1)/2,size(one_H,1)/6.5)); % normal distribution
    while dstart<1 || dstart>(size(one_H,1)-1)
        dstart=floor(normrnd(size(one_H,1)/2,size(one_H,1)/6.5)); % normal distribution
    end
    one_H=H(:,:,thiefs(i));
    [f_data, y, F_data,Y] = type1_2Fraud(one_H, intensity,dstart);
    F_data3D(:,:,thiefs(i))=F_data;
    Y2D(:,thiefs(i))=Y;
end
Y=(sum(Y2D)>0)';
% Details 
[kWh_count, time_count, kWh_rate, time_rate] = frauDetails(H, F_data3D);

daily_consumption=sum(F_data3D,2);
daily_consumption=permute(daily_consumption,[3 1 2]);
average_consumption=mean(daily_consumption,2);
%% Pick 4 Frauds
K=4;
idx=find(Y==1);
for i=1:K
    cluster_consumption(i,:)=daily_consumption(idx(i),:);
    sum_clusters(i)=1;
end
%% Fit quadratic trend
% Fit the second degree polynomial to the observed series.
tH=zeros(size(daily_consumption,2),K);
T = size(cluster_consumption,2);
t = (1:T)';
X = [ones(T,1) t t.^2];
figure
for i=1:K
    y=cluster_consumption(i,:)';
    b = X\y;
    tH(:,i) = X*b;

    
    subplot(2,2,i) % needs figure and hold off commented out
    plot(y/1000)
    h1 = gca;
    h1.XLim = [0,T];
    h1.XTick = 1:30:T;
    %h1.XTickLabel = datestr(dates(1:T),10);
    str=sprintf('Cluster %d\nAverage consumption of %d consumers',i ,sum_clusters(i));
    title(str);
    ylabel 'Consumption MWh';
    xlabel 'Days'
    hold on
    
    h2 = plot(tH(:,i)/1000,'r','LineWidth',2);
    legend(h2,'Quadratic Trend Estimate')
    
end
hold off
% Indicates when the minimum consumption happened
[min_tH, min_tH_idx]=min(tH,[],1);
%% Detrend  Original Series
% Subtract the fitted quadratic line from the original data.
xt=cluster_consumption -tH';
%% Estimate Seasonal Indicator Variables
% Create indicator (dummy) variables for each day of week/day of month. The first indicator is 
% equal to one for Monday observations, and zero otherwise. 
% The second indicator is equal to one for Tuesday observations, and zero otherwise. 
% A total of 7/12 indicator variables are created for the 7/12 weeks/months. 
% Regress the detrended series against the seasonal indicators.

prompt=('Choose Seasonality\n1 Day of Week, 0 Day of Month\n');
week=input(prompt);
% To get round values of weeks per year, get rid of one day.
if week==1
    st=zeros(364,K);
    one_less=randi(size(xt,2)); %day of week
    binary_remove=not(1:365==one_less);
    xt_less=xt(:,binary_remove);
else
% To get round values of months per year, get rid of 5 days.
    st=zeros(360,K);
    binary_remove=zeros(5,size(xt,2));
    some_less=randperm(size(xt,2),5); %day of month
    for i=1:size(some_less,2)
        binary_remove(i,:)=(1:365==some_less(i));
    end
    binary_remove=not(sum(binary_remove,1));
    xt_less=xt(:,binary_remove);
end

T_less=size(xt_less,2);
if week~=1
    figure
end
for i=1:K
    if week==1
        mo = repmat((1:7)',52,1); %day of week
    else
        mo=repmat((1:30)',12,1); %day of month
    end
    sX = dummyvar(mo);

    bS = sX\xt_less(i,:)';
    st(:,i) = sX*bS;

    if week==1
        figure
    else
        subplot(2,2,i)
    end
    plot(st(:,i)/1000)
    str=sprintf('Cluster %d\nParametric Estimate of Seasonal Component (Indicators)',i);
    title(str);
    h3 = gca;
    h3.XLim = [0,T_less];
    ylabel 'Consumption MWh';
    xlabel 'Days'
    if week==1
        h3.XTick = 1:7:T_less; %day of week
    else
        h3.XTick = 1:30:T_less; %day of week
    end
    %h3.XTickLabel = datestr(dates(1:12:T),10);
    if week==1
        hold off
    end
end
if week~=1
    hold off
end
[min_st,min_st_idx]=min(st,[],1);

%% Deseasonalize Original Series
% Subtract the estimated seasonal component from the original series.
% The quadratic trend is much clearer with the seasonal component removed.
    cluster_consumption_less=cluster_consumption(:,binary_remove);
    dt = cluster_consumption_less' - st;
figure
for i=1:K   
    
    subplot(2,2,i)
    plot(dt(:,i)/1000)
    str=sprintf('Cluster %d\nConsumption (Deseasonalized)',i);
    title(str);
    h4 = gca;
    h4.XLim = [0,T_less];
    ylabel 'Consumption MWh';
    xlabel 'Days'
    if week==1
        h4.XTick = 1:7:T_less;
    else
        h4.XTick = 1:30:T_less;
    end
    %h4.XTickLabel = datestr(dates(1:12:T),10);
end
hold off
%% Estimate Irregular Component
% Subtract the trend and seasonal estimates from the original series. 
% The remainder is an estimate of the irregular component.
tH_less=tH(binary_remove,:);
bt = cluster_consumption_less' - tH_less - st;
figure
for i=1:K
    
    subplot(2,2,i)
    plot(bt(:,i)/1000)
    str=sprintf('Cluster %d\nIrregular Component',i);
    title(str);
    h5 = gca;
    h5.XLim = [0,T_less];
    ylabel 'Consumption MWh';
    xlabel 'Days'
    if week==1
        h5.XTick = 1:7:T_less;
    else
        h5.XTick = 1:30:T_less;
    end
    %h5.XTickLabel = datestr(dates(1:12:T),10);
end
hold off