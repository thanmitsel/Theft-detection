function plotClass(X, Y)

figure; hold on;
% Find Indices of Positive Negative Examples
pos=find(Y==1); neg=find(Y==0);

% Plot Examples
plot(X(neg,1),X(neg,2),'ko','MarkerFaceColor','y','MarkerSize',7);
plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);

%title('Classified examples');
%legend('Negative','Positive');
title('Τάξεις των παραδειγμάτων');
legend('Αρνητική','Θετική');
hold off;
end