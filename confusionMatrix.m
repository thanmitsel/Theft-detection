function [prec,rec,acc,F1] = confusionMatrix (yval, pred)
    
tp=sum((pred==yval)&(pred==1)); % Predicted yes, actual yes
tn=sum((pred==yval)&(pred==0)); % Predicted no , actual no 
fp=sum((pred==1)&(yval==0)); % Predicted yes, actual no
fn=sum((pred==0)&(yval==1)); % Predicted no, actual yes 

prec=tp/(tp+fp); %
rec=tp/(tp+fn);
acc=((tp+tn)/(tp+tn+fp+fn));
F1=2*prec*rec/(prec+rec);
    

end
