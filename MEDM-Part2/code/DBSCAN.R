library(cluster)
library(dbscan)

##DBSCAN

#Heuristic for Minimum number of Points considered in a core point's neighbourhood
minPts <- ncol(X_train_norm) + 1

#Choose the value of Eps
diss_matrix_norm = daisy(X_train_norm, metric = "euclidean")
kNNdistplot(diss_matrix_norm, k = minPts-1)
abline(h = 0.3, lty = 2, col="blue") #0.3 eps
abline(h = 0.25, lty = 2, col="red") #0.25 eps
abline(h = 0.28, lty = 2, col="orange") #0.28 eps

#Obtain the clusters for the first "elbow"
db<- dbscan(diss_matrix_norm, 0.3, minPts)
db #2 clusters, but 34 noisy points

#Plot
fviz_cluster(db, dataset_norm_train[,-17], stand = FALSE, ellipse = FALSE, geom = "point")+
  scale_color_manual( values = c("royalblue4","royalblue1"))+
  scale_fill_manual(values= c("royalblue4","royalblue1"))


