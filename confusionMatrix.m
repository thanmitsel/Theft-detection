function [prec,rec, in_rec, acc,F1] = confusionMatrix (yval, pred)
    
tp=sum((pred==yval)&(pred==1)); % Predicted yes, actual yes
tn=sum((pred==yval)&(pred==0)); % Predicted no , actual no 
fp=sum((pred==1)&(yval==0)); % Predicted yes, actual no
fn=sum((pred==0)&(yval==1)); % Predicted no, actual yes 

prec=tp/(tp+fp)*100; 
rec=tp/(tp+fn)*100; % True Positive Rate-Detection Rate
in_rec=fp/(fp+tn)*100; % False Positive Rate
acc=((tp+tn)/(tp+tn+fp+fn))*100;
F1=2*prec*rec/(prec+rec); % Already a percent value
    

end
