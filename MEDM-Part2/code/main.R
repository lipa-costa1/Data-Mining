library(GGally)
library(ggplot2)
library(RColorBrewer)
library(ggsci)
library(wesanderson)
library(corrplot)
library(reshape)
library(sfsmisc)
library(varrank)
library(rrcov)
library(infotheo)
library(aricode)
library(robcor)
library(caret)
library(MASS)
library(psych)
library(readxl)

setwd("C:/Users/Filipa/Desktop/MEDM/MEDM-Grupo6-P2")
dataset <- read_excel("Dataset (1).xlsx")

### ================== Only Bombay and Dermason ==================== ####

dataset <- dataset[dataset$Class=="BOMBAY"|dataset$Class=="DERMASON",]
summary(dataset)

### ================== Train/Test split ==================== ####

nrows <- NROW(dataset)
set.seed(218)
index <- sample(1:nrows, 0.8 * nrows)

train <- dataset[index, ] # 3254 train data (80%)
test <- dataset[-index, ] # 814 test data (20%)

prop.table(table(train$Class)) # train set proportion of BOMBAY and DERMASON
prop.table(table(test$Class)) # test set proportion of BOMBAY and DERMASON

# Features
X_train <- train[-17]
X_test <- test[-17]
# Labels
y_train <- train$Class
y_test <- test$Class



### =================== Pre-processing ===================== ####

## Normalization - Min Max Scaler
min_max_scaling <- function(train, test) {
  min_vals <- sapply(train, min)
  range1 <- sapply(train, function(x) diff(range(x)))
  
  train_scaled <- data.frame(matrix(nrow = nrow(train), ncol = ncol(train)))
  
  for (i in seq_len(ncol(train))) {
    column <- (train[, i] - min_vals[i]) / range1[i]
    train_scaled[i] <- column
  }
  
  colnames(train_scaled) <- colnames(train)
  
  # scale the testing data using the min and range of the train data
  test_scaled <- data.frame(matrix(nrow = nrow(test), ncol = ncol(test)))
  
  for (i in seq_len(ncol(test))) {
    column <- (test[, i] - min_vals[i]) / range1[i]
    test_scaled[i] <- column
  }
  
  colnames(test_scaled) <- colnames(test)
  
  return(list(train = train_scaled, test = test_scaled))
}

norm <- min_max_scaling(X_train, X_test)

X_train_norm <- norm$train
X_test_norm <- norm$test


## Pearson Correlation for numerical variables - When deciding, remove the one
#that has a higher correlation with the other variables
set.seed(128)
correlationMatrix <- cor(X_train_norm)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.8)
X_train_corr_1 <- X_train_norm[, -highlyCorrelated]
X_test_corr_1 <- X_test_norm[, -highlyCorrelated]

## Pearson Correlation for numerical variables - When deciding, remove the one
#that has a lower correlation with the class variable

#this was done in the first part of the project. 
#Variables:EquivDiameter, Extent, Solidity, Roundness, Compactness and ShapeFactor4.

X_train_corr_2 <- X_train_norm[, c(8,9,10,11,12,16)]
X_test_corr_2 <- X_test_norm[, c(8,9,10,11,12,16)]

######Point-Bisserial Correlation between Numerical/Categorical

biserial(X_train_norm,y_train)#escolho só os que estao correlacionados com y acima de 0.95
highlyCorrelated95 <- c(1,2,3,4,7,8,13,14)#corresponde aqueles que têm uma 
#maior separação nos histogramas
biss<-X_train_norm[,c(1,2,3,4,7,8,13,14)]
biss_test<-X_test_norm[,c(1,2,3,4,7,8,13,14)]

ggpairs(biss)
correlationMatrix <- cor(biss)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.95)
X_train_biss <- biss[, -highlyCorrelated]
X_test_biss <- biss_test[, -highlyCorrelated]
cor(biss$ShapeFactor1,biss$ShapeFactor2)
#Only keep SF1 and SF2 variables

#Another dataset that was chosen in part1 - bisserial without removing highly 
#correlated
#Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea, EquivDiameter, ShapeFactor1
#and ShapeFactor2
X_train_cory<-X_train_norm[,c(1,2,3,4,7,8,13,14)]
X_test_cory<-X_test_norm[,c(1,2,3,4,7,8,13,14)]


#Different datasets: original, normalized,
#correlation pearson with average correlation,
#correlation pearson with highest with y, 
#bisserial between class, bisserial between class and remove highly correlated.

#Number of different datasets: 6



### ================== Datasets Export ===================== ####

path_out = './datasets'
dir.create(path_out)


datasets <- c('y_train', 
              'X_train_norm', 'X_train_corr_1','X_train_corr_2', 
              'X_train_biss', 'X_train_cory',
              
              'y_test', 
              'X_test_norm', 'X_test_corr_1','X_test_corr_2', 
              'X_test_biss', 'X_test_cory')


for (name in datasets) {
  
  data <- eval(as.name(name))
  
  if (substring(name, 1, 1) == "y") {data <- data.frame(Class = eval(as.name(name)))}
  
  write.csv(data, file.path(path_out, paste(name,'.csv', sep='')), row.names=FALSE)
}

