function [C, gamma] = findSVMparams (Xtrain, Ytrain, Xval, Yval)

C_test=[0.01 0.03 0.1 0.3 1 3 10 30];
gamma_test=[0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 30];

for i=1:length(C_test)
  for j=1:length(gamma_test)
        arguments=['-t ' num2str(2) ' -g ' num2str(gamma_test(j)) ' -c ' num2str(C_test(i))]; 
        model=svmtrain(Ytrain,Xtrain,arguments);
        predictions= svmpredict(Yval,Xval,model);
        error= mean(double(predictions ~= Yval));
        if i==1 && j==1
            min_error=error;
        end
        if error<min_error
            min_error=error;
            C=C_test(i);
            gamma=gamma_test(j);
        end
     end
 end
end

