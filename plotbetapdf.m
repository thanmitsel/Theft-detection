%% Plots beta pdf
X=0:.01:1;
y1=betapdf(X,2,4);
y2=betapdf(X,2,3);
y3=betapdf(X,2,50);
y4=betapdf(X,2,120);

figure;
plot(X,y1,'Color','r','LineWidth',2)
hold on
plot(X,y2,'LineStyle','-.','Color','b','LineWidth',2)
legend({'beta(2,4)','beta(2,3)'});
hold off

figure;
plot(X,y3,'Color','g','LineWidth',2)
hold on
plot(X,y4,'LineStyle',':','Color', 'm','LineWidth',2)
legend({'beta(2,50)','beta(2,120)'});
hold off