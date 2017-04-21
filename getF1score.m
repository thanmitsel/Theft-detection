function [F1]=getF1score(prediction, actual)

tp=sum((prediction==actual)&(prediction==1)); % Predicted yes, actual yes
tn=sum((prediction==actual)&(prediction==0)); % Predicted no , actual no 
fp=sum((prediction==1)&(actual==0)); % Predicted yes, actual no
fn=sum((prediction==0)&(actual==1)); % Predicted no, actual yes 


prec=tp/(tp+fp)*100; 
rec=tp/(tp+fn)*100; % True Positive Rate-Detection Rate
F1=2*prec*rec/(prec+rec); % Already a percent value