function [C, gamma] = naiveGridSearch (X,Y)
% Grid Search on C and gamma using cross-validation
K=5;
C_test=[2^(-5) 2^(-3) 2^(-1) 2^(1) 2^(3)...
    2^(5) 2^(7) 2^(9) 2^(11) 2^(13) 2^(15)];
gamma_test=[2^(-15) 2^(-13) 2^(-11) 2^(-9)...
    2^(-7) 2^(-5) 2^(-3) 2^(-1) 2^(1) 2^(3)];

min=10000;
for i=1:length(C_test)
  for j=1:length(gamma_test)
        gamma_temp=gamma_test(j);
        C_temp=C_test(i);
        indices=crossvalind('Kfold',size(X,1),K);
        error=0;
        for i=1:K
            Xval=X(indices==i,:);
            Yval=Y(indices==i);
            Xtrain=X(indices~=i,:);
            Ytrain=Y(indices~=i,:);
            arguments=['-t ' num2str(2) ' -g ' num2str(gamma_temp) ' -c ' num2str(C_temp)]; 
            model=svmtrain(Ytrain,Xtrain,arguments);
            predictions= svmpredict(Yval,Xval,model);
            temp_err=mean(double(predictions~= Yval));
            error=error+temp_err;
        end
        error=error/K; % mean of error on crossvalidation
        if (error < min)
            min=error;
            C=C_test(i);
            gamma=gamma_test(j);
        end

  end
end
end
