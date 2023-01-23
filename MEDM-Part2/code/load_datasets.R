# ------------------------------------------------------------- #
# Loads and stores the datasets as variables.                   #
# ------------------------------------------------------------- #

if (sys.nframe() == 0){
  path = './datasets'
}


files <- c('y_train', 
           'X_train_norm', 'X_train_corr_1','X_train_corr_2', 
           'X_train_biss', 'X_train_cory',
           
           'y_test', 
           'X_test_norm', 'X_test_corr_1','X_test_corr_2', 
           'X_test_biss', 'X_test_cory')


for (name in files) {
  assign(name, read.csv(file.path(path, paste(name,'.csv', sep=''))))
}