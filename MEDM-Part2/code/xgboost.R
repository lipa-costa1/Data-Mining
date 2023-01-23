library(modeldata)
library(dplyr)
library(fastDummies)
library(xgboost)
library(caret)
library(readr)
library(DiagrammeR)
library(SHAPforxgboost)


###########import datasets#############
dataset_biss_test_classifier <- read_csv("C:/Users/Catarina Rodrigues/Downloads/dataset_biss_test_classifier.csv")

dataset_biss_train_classifier <- read_csv("C:/Users/Catarina Rodrigues/Downloads/dataset_biss_train_classifier.csv")


################# as factor##########################3
dataset_biss_train_classifier$TrueClass<-as.factor(dataset_biss_train_classifier$TrueClass)
dataset_biss_train_classifier$ClusterClass<-as.factor(dataset_biss_train_classifier$ClusterClass)

dataset_biss_test_classifier$TrueClass<-as.factor(dataset_biss_test_classifier$TrueClass)
dataset_biss_test_classifier$ClusterClass<-as.factor(dataset_biss_test_classifier$ClusterClass)

####XGBOOst para TRUECLASS

dataset_biss_train_classifier_true<-dataset_biss_train_classifier[,-4]
dataset_biss_test_classifier_true<-dataset_biss_test_classifier[,-4]
#isolating y

train_y_true<-as.numeric(dataset_biss_train_classifier_true$TrueClass)
train_y_true <- as.numeric(dataset_biss_train_classifier_true$TrueClass)-1
test_y_true <- as.numeric(dataset_biss_test_classifier_true$TrueClass)-1


#isolating X

train_X_true<- dataset_biss_train_classifier_true %>% select(-TrueClass)

test_X_true <- dataset_biss_test_classifier_true %>% select(-TrueClass)

#checking the structure
str(train_X_true)

#setting the parameters

parameters <- list(set.seed=1999, eval_metric = "auc",objective = "binary:logistic")


#running xgboost

model_true <- xgboost(data=as.matrix(train_X_true),label = train_y_true,params = parameters,nrounds = 10,verbose = 1)

#evaluate model

predictions_true = predict(model_true,newdata=as.matrix(test_X_true))

predictions_true = ifelse(predictions_true>0.5,1,0)

#check the accuracy and more

confusionMatrix(table(predictions_true,test_y_true))


#look at the most important drivers

#shap values

# To return the SHAP values and ranked features by mean|SHAP|
shap_values_true <- shap.values(xgb_model = model_true, X_train =as.matrix(train_X_true))
# The ranked features by mean |SHAP|
shap_values_true$mean_shap_score

# To prepare the long-format data:
shap_long_true <- shap.prep(xgb_model = model_true, X_train = as.matrix(train_X_true))
shap_long_true <- shap.prep(shap_contrib = shap_values_true$shap_score, X_train = as.matrix(train_X_true))
shap.plot.summary(shap_long_true)


####XGBOOst para ClusterClass

dataset_biss_train_cluster<-dataset_biss_train_classifier[,-3]
dataset_biss_test_cluster<-dataset_biss_test_classifier[,-3]
#isolating y

train_y_cluster<-as.numeric(dataset_biss_train_cluster$ClusterClass)
train_y_cluster<- as.numeric(dataset_biss_train_cluster$ClusterClass)-1
test_y_cluster<- as.numeric(dataset_biss_test_cluster$ClusterClass)-1


#isolating X

train_X_cluster <- dataset_biss_train_cluster %>% select(-ClusterClass)

test_X_cluster <- dataset_biss_test_cluster %>% select(-ClusterClass)

#checking the structure
str(train_X_cluster)

#setting the parameters

parameters <- list(set.seed=1999, eval_metric = "auc",objective = "binary:logistic")


#running xgboost

model_cluster <- xgboost(data=as.matrix(train_X_cluster),label = train_y_cluster,params = parameters,nrounds = 10,verbose = 1)

#evaluate model

predictions_cluster = predict(model_cluster,newdata=as.matrix(test_X_cluster))

predictions_cluster = ifelse(predictions_cluster>0.5,1,0)

#check the accuracy and more

confusionMatrix(table(predictions_cluster,test_y_cluster))

#look at the most important drivers

#shap values

# To return the SHAP values and ranked features by mean|SHAP|
shap_values_cluster <- shap.values(xgb_model = model_cluster, X_train =as.matrix(train_X_cluster))
# The ranked features by mean |SHAP|
shap_values_cluster$mean_shap_score

# To prepare the long-format data:
shap_long_cluster <- shap.prep(xgb_model = model_cluster, X_train = as.matrix(train_X_cluster))
shap_long_cluster <- shap.prep(shap_contrib = shap_values_cluster$shap_score, X_train = as.matrix(train_X_cluster))
shap.plot.summary(shap_long_cluster)


