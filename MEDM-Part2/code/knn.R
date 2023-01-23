library(caret)
library(readr)
library(kknn)

###########import datasets#############
dataset_biss_test_classifier <- read_csv("C:/Users/Catarina Rodrigues/Downloads/dataset_biss_test_classifier.csv")

dataset_biss_train_classifier <- read_csv("C:/Users/Catarina Rodrigues/Downloads/dataset_biss_train_classifier.csv")

# plot to see which is the best k value for Clustering class with manhattan metric
grid <- expand.grid(k = c(1:10))
ClusterClass_knn <- train(ClusterClass ~., method= "knn",
                       data =  dataset_biss_train_classifier[,-3],
                       trControl = trainControl(method = 'cv',
                                                number = 3,
                                                search = "grid"),
                       tuneGrid = grid)



plot(ClusterClass_knn,xlim=c(0,10),ylim=c(0,1.03))

# plot to see which is the best k value for True Class with manhattan metric
grid <- expand.grid(k = c(1:10))
TrueClass_knn <- train(TrueClass ~., method= "knn",
                    data =  dataset_biss_train_classifier[,-4],
                    trControl = trainControl(method = 'cv',
                                             number = 3,
                                             search = "grid"),
                    tuneGrid = grid)



plot(TrueClass_knn,xlim=c(0,10),ylim=c(0,1.03))
################# as factor##########################
dataset_biss_train_classifier$TrueClass<-as.factor(dataset_biss_train_classifier$TrueClass)
dataset_biss_train_classifier$ClusterClass<-as.factor(dataset_biss_train_classifier$ClusterClass)

dataset_biss_test_classifier$TrueClass<-as.factor(dataset_biss_test_classifier$TrueClass)
dataset_biss_test_classifier$ClusterClass<-as.factor(dataset_biss_test_classifier$ClusterClass)

#######KNN for predicting TrueClass, with k=1 and p=1(Manhattan)
knn.1.true <- kknn(formula = formula(TrueClass~.), train = dataset_biss_train_classifier[,-4], test = dataset_biss_test_classifier[,-4], k = 1, distance = 1)

#######prediction####
#confusion matrix
fit <- fitted(knn.1.true) 
table(dataset_biss_test_classifier$TrueClass, fit)


performance_meas <- function(predicted, real) {
  
  aux <- confusionMatrix(data = predicted, reference = real)
  
  print(aux$table)
  
  return (data.frame(
    value = c(aux$overall['Accuracy'], aux$byClass['Sensitivity'],
              aux$byClass['Specificity'], aux$byClass['Balanced Accuracy'],
              aux$byClass['Precision'], aux$byClass['F1'])))
}


performance_meas(as.factor(fit), dataset_biss_test_classifier$TrueClass)




#######KNN for predicting ClusterClass, with k=1 and p=1(Manhattan)
knn.1.cluster <- kknn(formula = formula(ClusterClass~.), train = dataset_biss_train_classifier[,-3], test = dataset_biss_test_classifier[,-3], k = 1, distance = 1)

#######prediction####
#confusion matrix
fit.cluster <- fitted(knn.1.cluster)
table(dataset_biss_test_classifier$ClusterClass, fit.cluster)


performance_meas(as.factor(fit.cluster), dataset_biss_test_classifier$ClusterClass)
