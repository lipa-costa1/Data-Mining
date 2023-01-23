library(cluster)
library(factoextra)
library(clusterCrit)
library(tidyr)
library(caret)
library(mlbench)
library(ClusterR)


#### K-Means
###################LOADING DATASETS###################
  
dataset_norm_train<-cbind(X_train_norm,y_train)
dataset_norm_test<-cbind(X_test_norm,y_test)

dataset_corr_1_train<-cbind(X_train_corr_1,y_train)
dataset_corr_1_test<-cbind(X_test_corr_1,y_test)

dataset_corr_2_train<-cbind(X_train_corr_2,y_train)
dataset_corr_2_test<-cbind(X_test_corr_2,y_test)

dataset_biss_train<-cbind(X_train_biss,y_train)
dataset_biss_test<-cbind(X_test_biss,y_test)

dataset_cory_train<-cbind(X_train_cory,y_train)
dataset_cory_test<-cbind(X_test_cory,y_test)


#######NORM########
results_means(dataset_norm_train[,-17],dataset_norm_train, dataset_norm_test[,-17], dataset_norm_test)

######CORR 1########
results_means(dataset_corr_1_train[,-7],dataset_corr_1_train, dataset_corr_1_test[,-7], dataset_corr_1_test)

########CORR 2###
results_means(dataset_corr_2_train[,-7],dataset_corr_2_train, dataset_corr_2_test[,-7], dataset_corr_2_test)
#1 mal classificada

############BISS###
results_means(dataset_biss_train[,-3],dataset_biss_train, dataset_biss_test[,-3], dataset_biss_test)


###########CORY###
results_means(dataset_cory_train[,-9],dataset_cory_train, dataset_cory_test[,-9], dataset_cory_test)


###########Deram todos os mesmos resultados. 
####Logo o melhor dataset é o #biss, só com 2 variáveis.


##############Melhor modelo -- Bisserial entre X e Y
set.seed(128)
km<-kmeans(dataset_biss_train[,-3],centers=2,nstart=30)
#visualize the clusters
data <- data.frame(X_train_biss,km$cluster)
ggplot(data, aes(x = X_train_biss[,1], y = X_train_biss[,2], color = km$cluster))+ geom_point()+
  ggtitle("Cluster Plot K-Means") +
  theme_bw() +
  labs(
    x = "Shape Factor 1",
    y = "Shape Factor 2"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 12, face = "bold"), plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.title.x = element_text(face = "bold"), axis.title.y = element_text(face = "bold")
  )



######Elbow Method

elbow(dataset_biss_train[,-3])

ggplot(elbow(dataset_biss_train[,-3]), aes(numero_clusters, error)) + geom_line(colour = "cornflowerblue") + geom_point() + 
  theme_minimal() + scale_x_continuous(breaks = seq(1,30))+
  ggtitle("Total Within-Cluster Sum of Squares by number of cluster") +
  theme_bw() +
  labs(
    x = "Number of Clusters",
    y = "Total Within-Cluster Sum of Squares"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 12, face = "bold"), plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.title.x = element_text(face = "bold"), axis.title.y = element_text(face = "bold")
  )

#Dendogram
a4_eu_origi<-agnes(dataset_biss_train[,-3], metric = "euclidean",
                  stand = FALSE, method = "ward", keep.data = FALSE)
fviz_dend(a4_eu_origi, k = 2,cex = 0.5,k_colors = c("royalblue4","royalblue1"),
          color_labels_by_k = TRUE, ggtheme = theme_minimal())

#Dark <- DERMASON
#Light <- BOMBAY

pltree(a4_eu_origi,main="Ward Method", cex=0.83,xlab="")
rect.hclust(a4_eu_origi, k=2, border="royalblue1")



#########################Final Dataset for classification

######Vamos aplicar o cluster ao dataset todo.
#Scale dataset todo:
min_max_scaling_mod <- function(train) {
  min_vals <- sapply(train, min)
  range1 <- sapply(train, function(x) diff(range(x)))
  
  train_scaled <- data.frame(matrix(nrow = nrow(train), ncol = ncol(train)))
  
  for (i in seq_len(ncol(train))) {
    column <- (train[, i] - min_vals[i]) / range1[i]
    train_scaled[i] <- column
  }
  
  colnames(train_scaled) <- colnames(train)
  return(train = train_scaled)}

dataset_biss<-dataset[,c(13,14,17)]
dataset_biss<-cbind(min_max_scaling_mod(dataset_biss[,-3]),dataset_biss[,3])

#cluster dataset todo
km<-kmeans(dataset_biss[,-3],centers=2,nstart=30)

#Metemos as labels por clustering no dataset.
dataset_biss<-cbind(dataset_biss,factor(ifelse(km$cluster == 1,"DERMASON","BOMBAY")))
names(dataset_biss)<-c("ShapeFactor1","ShapeFactor2","TrueClass", "ClusterClass")

#Dividimos novamente em treino e teste.
nrows <- NROW(dataset_biss)
set.seed(218)
index <- sample(1:nrows, 0.8 * nrows)

dataset_biss_train_classifier <- dataset_biss[index, ] # 3254 train data (80%)
dataset_biss_test_classifier <- dataset_biss[-index, ] # 814 test data (20%)

##Exportar

path_out = './datasets'

datasets <- c('dataset_biss_train_classifier', 'dataset_biss_test_classifier')


for (name in datasets) {
  
  data <- eval(as.name(name))
  
  write.csv(data, file.path(path_out, paste(name,'.csv', sep='')), row.names=FALSE)
}

