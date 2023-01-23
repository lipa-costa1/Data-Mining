##Graph-Based Clustering
library(mstknnclust)
library(cluster)
library(igraph)

diss_matrix_norm = daisy(X_train_norm, metric = "euclidean")
graph_clust <- mstknnclust::mst.knn(as.matrix(diss_matrix_norm))
table(graph_clust$cluster)
#37 clusters


###########################

#há mais do que dois clusters logo há diversos grupos 
#nos dados, não apenas característicos de feijões BOMBAY ou DERMASON
