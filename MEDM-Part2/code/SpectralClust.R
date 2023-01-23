##Spectral Clustering
library(Spectrum) 
mat<-as.data.frame(t(X_train_norm))
spec <- Spectrum::Spectrum(mat)
spec$K #3 clusters

table(spec$assignments)
table(dataset_norm_train$Class)
#diria que DERMASON foi mal classificado 

##Plot clusters
library(factoextra)
fviz_cluster(list(data = X_train_norm, cluster = spec$assignments), stand = T) + ggtitle("Clusters")
