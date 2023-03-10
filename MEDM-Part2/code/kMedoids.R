library(factoextra)
library(cluster)
library(clusterCrit)
library(tidyr)
library(caret)
library(NMF)

######LOADING DATASETS
dataset_norm_train<-cbind(X_train_norm,y_train)
dataset_norm_test<-cbind(X_test_norm,y_test)


#Performance measures
#(the predicted and real must in in "BOMBAY" and "DERMASON" factors)
performance_meas<-function(predicted,real){
  
  performance_m = data.frame(
    medidas = c("Accuracy", "Sensitivity/Recall", "Specificity", "Balanced Accuracy",
                "Precision",  "F1-score"),
    valor = NA
  )
  aux<-confusionMatrix(data=predicted,reference=real)
  performance_m$valor[1]<-aux$overall[1]
  performance_m$valor[2]<-aux$byClass[1]
  performance_m$valor[3]<-aux$byClass[2]
  performance_m$valor[4]<-aux$byClass[11]
  performance_m$valor[5]<-precision(data=predicted,reference=real)
  performance_m$valor[6]<-aux$byClass[7]
  return(performance_m)
}


#####Purity function

ClusterPurity <- function(clusters, classes) {
  sum(apply(table(classes, clusters), 2, max)) / length(clusters)
}


######Silhouette Method and Other Methods - Davies Bound-index, Dunn-index, C-index
best_cluster_eu<- function(df) {
  
  ss=c()
  scorefunc = data.frame(
    numero_clusters = c(2:7),
    scores = NA,
    cs = NA,
    dbs = NA,
    dunns = NA
    
  )
  
  for(i in 1:6){
    km <- pam(df, i+1, metric = 'euclidean', stand = FALSE)
    ss <- silhouette(km$clustering,dist = dist(df))
    scorefunc$scores[i] = mean(ss[,3])
    aux <- intCriteria(as.matrix(df),km$clustering,c("C_index","Davies_Bouldin","Dunn"))
    scorefunc$cs[i]<-aux$c_index
    scorefunc$dbs[i]<-aux$davies_bouldin
    scorefunc$dunns[i]<-aux$dunn
  }
  
  return(scorefunc)
  
}
best_cluster_ma<- function(df) {
  
  ss=c()
  scorefunc = data.frame(
    numero_clusters = c(2:7),
    scores = NA,
    cs = NA,
    dbs = NA,
    dunns = NA
    
  )
  
  for(i in 1:6){
    km <- pam(df, i+1, metric = 'manhattan', stand = FALSE)
    ss <- silhouette(km$clustering,dist = dist(df))
    scorefunc$scores[i] = mean(ss[,3])
    aux <- intCriteria(as.matrix(df),km$clustering,c("C_index","Davies_Bouldin","Dunn"))
    scorefunc$cs[i]<-aux$c_index
    scorefunc$dbs[i]<-aux$davies_bouldin
    scorefunc$dunns[i]<-aux$dunn
  }
  
  return(scorefunc)
  
}

############FUNCTIONS FOR PREDICT
#euclidian distance

predict.pam <- function(object, newdata){
  centers <- object$medoids
  n_centers <- nrow(centers)
  dist_mat <- as.matrix(dist(rbind(centers, newdata)))
  dist_mat <- dist_mat[-seq(n_centers), seq(n_centers)]
  max.col(-dist_mat)
}

#pam which stands for "partitioning around medoids" and uses the following syntax:
pam_ma<-pam(X_train_norm,2, metric = 'manhattan', stand = FALSE)
pam_eu<-pam(X_train_norm, 2, metric = 'euclidean', stand = FALSE)


#Usando o metodo silhouette para obter o melhor cluster
best_cluster_eu(X_train_norm)
best_cluster_ma(X_train_norm)


#ELBOW METHOD com metric=euclidian function to create a plot of the number of clusters vs. the total within sum of squares:
fviz_nbclust(X_train_norm, pam, method = "wss")

#tambem da para confirmar que o k=2 ? onde ocorre o elbow


set.seed(128)
#k-medoids with k=2 clusters and metric euclidean
kmedoids <- pam(X_train_norm, k=2)
kmedoids # para ver os resultados


#visualize the clusters
#plot results of final k-medoids model
fviz_cluster(kmedoids, data = X_train_norm)


########EVALUATE THE PREVIS?O with k=2

#Supervised or external indices: 
#measures how coherent the partition is with external information about previous
#known classes of objects.

results_medoids<-function(df_train_semy,df_train_comy,df_test_semy,df_test_comy){

set.seed(47)
kmedoids<-pam(df_train_semy, k=2)

#Performance Measures - In the training set

pre_kmedoids <- kmedoids$clustering
pre_kmedoids <- factor(ifelse(pre_kmedoids == 1,"DERMASON","BOMBAY")) #1-DERMASON, 2-BOMBAY
yy_train <- factor(df_train_comy[,length(df_train_comy)])
print("Performance Measures - In the training set")
print(performance_meas(pre_kmedoids,yy_train))
print(confusionMatrix(data=pre_kmedoids,reference=yy_train)$table)

r <- rbind(
  kmeans = c(
    purity=ClusterPurity(pre_kmedoids,yy_train),
    entropy=entropy(pre_kmedoids,yy_train)$U
  )
)

print(r)

#Performance Measures - In the test set
#k-medoids with k=2 clusters and metric euclidean

test_preds <- predict(kmedoids, df_test_semy)
test_preds <- factor(ifelse(test_preds == 1,"DERMASON","BOMBAY")) #1-DERMASON, 2-BOMBAY
yy_test <- factor(df_test_comy[,length(df_train_comy)])
print("Performance Measures - In the test set")
print(performance_meas(test_preds,yy_test))
print(confusionMatrix(data=test_preds,reference=yy_test)$table)



}

results_medoids(dataset_norm_train[,-17],dataset_norm_train, dataset_norm_test[,-17], dataset_norm_test)


#Mesmos resultados que HC e kMeans

