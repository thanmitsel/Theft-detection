function plotElbow(clusters, cost)
figure;
plot(clusters,cost);
str = sprintf('Elbow Method');
title(str);
xlabel('Clusters');
ylabel('Cost Function');
end