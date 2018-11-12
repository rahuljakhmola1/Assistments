#import data
test_data <- read.csv("labelled_training_data.csv")

#generate input features
input_test <- features_1(test_data)

#best model predictions
results <-predict(ridge_model, s=best_ridge_lambda, as.matrix(input_test[,-(1:2)]), type="response")
ridge_con <- confusionMatrix(as.factor(con.valid[,2]), as.factor(ifelse(results>0.5, 1, 0)))
ridge_con$overall["Accuracy"]

# Install the ROCR package
library('ROCR')

# AUC function
fun.auc <- function(pred,obs){
  # Run the ROCR functions for AUC calculation
  ROC_perf <- performance(prediction(pred,obs),"tpr","fpr")
  ROC_sens <- performance(prediction(pred,obs),"sens","spec")
  ROC_err <- performance(prediction(pred, labels=obs),"err")
  ROC_auc <- performance(prediction(pred,obs),"auc")
  # AUC value
  AUC <- ROC_auc@y.values[[1]] # AUC
  x.Sens <- mean(as.data.frame(ROC_sens@y.values)[,1])
  x.Spec <- mean(as.data.frame(ROC_sens@x.values)[,1])
  SS <- data.frame(SENS=as.data.frame(ROC_sens@y.values)[,1],SPEC=as.data.frame(ROC_sens@x.values)[,1])
  SS_min_dif <- ROC_perf@alpha.values[[1]][which.min(abs(SS$SENS-SS$SPEC))]
  SS_max_sum <- ROC_perf@alpha.values[[1]][which.max(rowSums(SS[c("SENS","SPEC")]))]
  Min_Err <- min(ROC_err@y.values[[1]])
  Min_Err_Cut <- ROC_err@x.values[[1]][which(ROC_err@y.values[[1]]==Min_Err)][1]
  round(cbind(AUC,x.Sens,x.Spec,SS_min_dif,SS_max_sum,Min_Err,Min_Err_Cut),3)
}

# Run the function with the example data
AUC <- fun.auc(ifelse(results>0.5, 1, 0), input_test[,2])


## RMSE Calculation
predictions <- ifelse(results>0.5, 1, 0)
observations <- input_test[,2]

RMSE = (sum((predictions - observations)^2)/nrow(input_test))^0.5
RMSE
unique(AUC[AUC])
1-RMSE+unique(AUC[AUC])
