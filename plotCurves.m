function plotCurves(X,Y)
% This funstion is used to plot ROC curves
% X and Y should be matrices 
% X coordinate is FPR-inverse recall
% Y coordinate is DR-recall
plot(X(:,1),Y(:,1),'color', 'r'); hold on;
plot(X(:,2),Y(:,2),'color','b'); hold on;
plot(X(:,3),Y(:,3),'color','g');
xlabel('False Positive Rate');
ylabel('Detection Rate');
legend('High','Medium','Low');
end