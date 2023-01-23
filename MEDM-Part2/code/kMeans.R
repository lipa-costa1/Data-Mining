#### K-Means
library(cluster)
library(factoextra)
library(clusterCrit)
library(tidyr)
library(caret)
library(mlbench)
library(ClusterR)

############FUNCTIONS FOR CLUSTERING VALIDATION 
#(the df is the dataset without the class column)
######Elbow Method

elbow<-function(df){
  
  errors = data.frame(
    numero_clusters = seq(1:7),
    error = NA
  )
  
  for(i in 1:7){
    cluster <-  kmeans(df, centers = i, nstart = 30)
    
    errors$error[i] = cluster$tot.withinss
  }
  return(errors)
}

######Silhouette Method and Other Methods - Davies Bound-index, Dunn-index, C-index


best_cluster <- function(df) {
  
  ss=c()
  scorefunc = data.frame(
    numero_clusters = c(2:7),
    scores = NA,
    cs = NA,
    dbs = NA,
    dunns = NA
    
  )
  
  for(i in 1:6){
    km <-  kmeans(df, centers = i+1, nstart = 30)
    ss <- silhouette(km$cluster,dist = dist(df))
    scorefunc$scores[i] = mean(ss[,3])
    aux <- intCriteria(as.matrix(df),km$cluster,c("C_index","Davies_Bouldin","Dunn"))
    scorefunc$cs[i]<-aux$c_index
    scorefunc$dbs[i]<-aux$davies_bouldin
    scorefunc$dunns[i]<-aux$dunn
  }
  
  return(scorefunc)
  
}

###### MINIMIZAR A WITHIN E MAXIMIZAR A BETWEEN Method


within_between<-function(df){
  
  ssc = data.frame(
    numero_clusters = seq(1:7),
    within_ss = NA,
    between_ss = NA
  )
  
  for(i in 1:7){
    cluster <-  kmeans(df, centers = i, nstart = 30)
    
    ssc$within_ss[i] = mean(cluster$withinss)
    ssc$between_ss[i] = cluster$betweenss
  }
  return(ssc)
}


############FUNCTIONS FOR PERFORMANCE MEASURES
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

########

############FUNCTIONS FOR PREDICT
#euclidian distance

predict.kmeans <- function(object, newdata){
  centers <- object$centers
  n_centers <- nrow(centers)
  dist_mat <- as.matrix(dist(rbind(centers, newdata)))
  dist_mat <- dist_mat[-seq(n_centers), seq(n_centers)]
  max.col(-dist_mat)
}


#############


######LOADING DATASETS
dataset_norm_train<-cbind(X_train_norm,y_train)
dataset_norm_test<-cbind(X_test_norm,y_test)


#####Choosing the optimal k

#One way of determing the number is by using the elbow method.
#This method is based on running several K-means algorithm with different ks.
#If we plot the overall error for the different number of clusters we will see 
#that the more clusters that we add to the algorithm the 
#lower that the total error will be.
#However, the error will know the decrease in a uniformed way: 
#we will reach one point where the error does not decrease as much
#as it does on the previous step. That is the elbow of the graph and
#that's the number of clusters that we should choose.


######Elbow Method

elbow(dataset_norm_train[,-17])

ggplot(elbow(dataset_norm_train[,-17]), aes(numero_clusters, error)) + geom_line(colour = "cornflowerblue") + geom_point() + 
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


#As we can see, the error decreases a lot when we pass from 1 cluster to 
#two.
#However, the total error does not decrease that much when we add a third cluster.
#So, we should look for two clusters on our dataset.


######Silhouette Method and Other Methods - Davies Bound-index, Dunn-index, C-index

#Average silhouette method computes the average silhouette of observations
#for different values of k. The optimal number of clusters k is the one
#that maximize the average silhouette over a range of possible values for k.


best_cluster(dataset_norm_train[,-17])
plot(c(2:7), type='b', best_cluster(dataset_norm_train[,-17])$scores, xlab='Number of clusters', ylab='Average Silhouette Width', frame=FALSE)

######Minimizar a Within e Maximizar a Between Method

f<-gather(within_between(dataset_norm_train[,-17]), key = "measurement", value = value,-numero_clusters)
ggplot(f, aes(x=numero_clusters, y=value, fill = measurement)) +
geom_bar(stat = "identity", position = "dodge") + 
scale_x_discrete(name = "Number of Clusters", limits = c("1", "2", "3", "4", "5", "6", "7"))+
  theme_minimal() +
  ggtitle("Cluster Model Comparison") +
  theme_bw() +
  labs(
    x = "Number of Clusters",
    y = "Total Sum of Squares"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 12, face = "bold"), plot.subtitle = element_text(hjust = 0.5, size = 10),
    axis.title.x = element_text(face = "bold"), axis.title.y = element_text(face = "bold")
  )


#Meter plot no relatorio 

###Seems like k=2

########EVALUATE THE PREVISÃO with k=2


results_means<-function(df_train_semy,df_train_comy,df_test_semy,df_test_comy){
  
  set.seed(47)
  km2<-kmeans(df_train_semy, centers = 2, nstart = 30)
  
  #Performance Measures - In the training set
  
  pre_kmeans <- km2$cluster
  pre_kmeans <- factor(ifelse(km2$cluster == 1,"DERMASON","BOMBAY")) #1-DERMASON, 2-BOMBAY
  yy_train <- factor(df_train_comy[,length(df_train_comy)])
  print("Performance Measures - In the training set")
  print(performance_meas(pre_kmeans,yy_train))
  print(confusionMatrix(data=pre_kmeans,reference=yy_train)$table)
  
  r <- rbind(
    kmeans = c(
      purity=ClusterPurity(pre_kmeans,yy_train),
      entropy=entropy(pre_kmeans,yy_train)$U
    )
  )
  
  print(r)
  
  #Performance Measures - In the test set
  
  test_preds <- predict(km2, df_test_semy)
  test_preds <- factor(ifelse(test_preds == 1,"DERMASON","BOMBAY")) #1-DERMASON, 2-BOMBAY
  yy_test <- factor(df_test_comy[,length(df_train_comy)])
  print("Performance Measures - In the test set")
  print(performance_meas(test_preds,yy_test))
  print(confusionMatrix(data=test_preds,reference=yy_test)$table)

  
}

results_means(dataset_norm_train[,-17],dataset_norm_train, dataset_norm_test[,-17], dataset_norm_test)

##Resultados iguais ao HC

