function [C, gamma] = findSVMparams (Xtrain, Ytrain, Xval, Yval)

C_test=[0.01 0.03 0.1 0.3 1 3 10 30];
gamma_test=[0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30];

min=10000;
for i=1:length(C_test)
  for j=1:length(gamma_test)
        gamma_temp=gamma_test(j);
        C_temp=C_test(i);
        arguments=['-t ' num2str(2) ' -g ' num2str(gamma_temp) ' -c ' num2str(C_temp)]; 
        model=svmtrain(Ytrain,Xtrain,arguments);
        predictions= svmpredict(Yval,Xval,model);
        error=mean(double(predictions~= Yval));
        if (error < min)
            min=error;
            C=C_test(i);
            gamma=gamma_test(j);
        end

  end
end
end

