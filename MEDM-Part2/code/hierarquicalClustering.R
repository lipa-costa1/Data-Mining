#### Hierarquical Clustering
library(cluster)
library(fpc)
library(factoextra)
library(clusterCrit)
library(tidyr)
library(caret)

##### Functions for find Optimal number of clusters

### Average Silhouette Width
# hc= Single, Complete, Average, Ward

silhouettee<-function(datatrain_sem_y,hc){
  ks <- 2:20
  ASW <- sapply(ks, FUN=function(k) {
    fpc::cluster.stats(dist(datatrain_sem_y), cutree(hc, k))$avg.silwidth
  })
  plot(ks, ASW, type="l")
  abline(v=ks[which.max(ASW)], col="red", lty=2)
  print(ks[which.max(ASW)])
}

### Other Methods - Davies Bound-index, Dunn-index, C-index

#df=a1_eu_orig

best_cluster_hc <- function(df_sem_y,df) {
  
  scorefunc = data.frame(
    numero_clusters = c(2:7),
    cs = NA,
    dbs = NA,
    dunns = NA
    
  )
  
  for(i in 1:6){
    hc <-  cutree(df,i+1)
    aux <- intCriteria(as.matrix(df_sem_y),hc,c("C_index","Davies_Bouldin","Dunn"))
    scorefunc$cs[i]<-aux$c_index
    scorefunc$dbs[i]<-aux$davies_bouldin
    scorefunc$dunns[i]<-aux$dunn
  }
  
  return(scorefunc)
  
}


##### FUNCTIONS FOR PERFORMANCE MEASURES
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

##### FUNCTIONS FOR PREDICT
#euclidian distance

predicthc <- function(object, newdata, df_train_semy){
  centers <- aggregate(df_train_semy,list(cluster=object),mean)[,-1]
  n_centers <- nrow(centers)
  dist_mat <- as.matrix(dist(rbind(centers, newdata)))
  dist_mat <- dist_mat[-seq(n_centers), seq(n_centers)]
  max.col(-dist_mat)
}

#############


####Loading datasets
dataset_norm_train<-cbind(X_train_norm,y_train)
dataset_norm_test<-cbind(X_test_norm,y_test)


######Hierarchical Clustering

a1_eu_orig<-agnes(dataset_norm_train[,-17], metric = "euclidean",
                  stand = FALSE, method = "single", keep.data = FALSE)
a2_eu_orig<-agnes(dataset_norm_train[,-17], metric = "euclidean",
                  stand = FALSE, method = "complete", keep.data = FALSE)
a3_eu_orig<-agnes(dataset_norm_train[,-17], metric = "euclidean",
                  stand = FALSE, method = "average", keep.data = FALSE)
a4_eu_orig<-agnes(dataset_norm_train[,-17], metric = "euclidean",
                  stand = FALSE, method = "ward", keep.data = FALSE)
par(mfrow=c(1,1))
pltree(a1_eu_orig,main="Single linkage", cex=0.83,xlab="")
pltree(a2_eu_orig,main="Complete linkage",cex=0.83,xlab="")
pltree(a3_eu_orig,main="Average linkage", cex=0.83,xlab="")
pltree(a4_eu_orig,main="Ward Method", cex=0.83,xlab="")

##Find optimal clusters 

#Single Linkage
silhouettee(dataset_norm_train[,-17],a1_eu_orig)
best_cluster_hc(dataset_norm_train[,-17],a1_eu_orig)
#Complete Linkage
silhouettee(dataset_norm_train[,-17],a2_eu_orig)
best_cluster_hc(dataset_norm_train[,-17],a2_eu_orig)
#Average Linkage
silhouettee(dataset_norm_train[,-17],a3_eu_orig)
best_cluster_hc(dataset_norm_train[,-17],a3_eu_orig)
#Ward Method
silhouettee(dataset_norm_train[,-17],a4_eu_orig)
best_cluster_hc(dataset_norm_train[,-17],a4_eu_orig)

#dbs ? o menor. dunn ? o maior. 


#no single linkage e average linkage os indices nao coincidem.
#temos de escolher entre complete linkage e ward method. 
#Pelos gr?ficos do Average Silhouette Width, aquele que apresenta uma maior 
#queda entre k=2 e k=3 ? o ward method. 
#Escolhemos o Ward method. 

##Seems like 2 clusters

#Dendrogram with red lines surrounding the 2 clusters, with Ward Method

pltree(a4_eu_orig,main="Ward Method", cex=0.83,xlab="")
rect.hclust(a4_eu_orig, k=2, border="royalblue1")

#Prettiest Dendrogram

fviz_dend(a4_eu_orig, k = 2,cex = 0.5,k_colors = c("royalblue4","royalblue1"),
          color_labels_by_k = TRUE, ggtheme = theme_minimal())

#Dark <- DERMASON
#Light <- BOMBAY

#Cutting the tree of the Hierarchical in k=2 clusters
ca1_eu_orig<-cutree(a1_eu_orig,2)
ca2_eu_orig<-cutree(a2_eu_orig,2)
ca3_eu_orig<-cutree(a3_eu_orig,2)
ca4_eu_orig<-cutree(a4_eu_orig,2)

#Confusion matrix and Performance Measures

#With Ward Method
results_hc<-function(df_train_semy,df_train_comy,df_test_semy,df_test_comy){
  
  set.seed(47)
  a_eu_orig<-agnes(df_train_semy, metric = "euclidean",
                    stand = FALSE, method = "ward", keep.data = FALSE)
  ca_eu_orig<-cutree(a_eu_orig,2)
  
  #Performance Measures - In the training set
  
  pre_hc <- ca_eu_orig
  pre_hc <- factor(ifelse(ca_eu_orig == 1,"DERMASON","BOMBAY")) #1-DERMASON, 2-BOMBAY
  yy_train <- factor(df_train_comy[,length(df_train_comy)])
  print("Performance Measures - In the training set")
  print(performance_meas(pre_hc,yy_train))
  print(confusionMatrix(data=pre_hc,reference=yy_train)$table)
  r <- rbind(
    hc = c(
      purity=ClusterPurity(pre_hc,yy_train),
      entropy=entropy(pre_hc,yy_train)$U
    )
  )
  
  print(r)
  
  #Performance Measures - In the test set
  
  test_preds <- predicthc(ca_eu_orig, df_test_semy, df_train_semy)
  test_preds <- factor(ifelse(test_preds == 1,"DERMASON","BOMBAY")) #1-DERMASON, 2-BOMBAY
  yy_test <- factor(df_test_comy[,length(df_train_comy)])
  print("Performance Measures - In the test set")
  print(performance_meas(test_preds,yy_test))
  print(confusionMatrix(data=test_preds,reference=yy_test)$table)
  

  
}


results_hc(dataset_norm_train[,-17],dataset_norm_train, dataset_norm_test[,-17], dataset_norm_test)




