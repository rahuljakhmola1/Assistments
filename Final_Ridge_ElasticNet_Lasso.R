##Final_Ridge_ElasticNet_Lasso

###https://www.youtube.com/watch?v=FWCPFUwZkn0


library(MASS)
library(caret)
library(rpart)
library(rpart.plot)
library(FNN)

## Importing the file
df <- read.csv("labelled_training_data.csv")
df$X <- NULL

## creating pivots
library(data.table)
df_piv1 <- features_1(df)

# Splitting the condensed emo file into training and validation data set
set.seed(123)
training.index <- createDataPartition(df_piv1$ITEST_id, p = 0.8, list = FALSE)
con.train <- df_piv1[training.index, ]
con.valid <- df_piv1[-training.index, ]

##fit a lasso model
library(glmnet)
set.seed(123)
lasso_model <- cv.glmnet(as.matrix(con.train[,-(1:2)]), as.matrix(con.train[,2]),
                         lambda = 10^seq(4,-1,-1), alpha=1)
#lasso_model$lambda.1se
#plot(lasso_model)
best_lasso_lambda <- lasso_model$lambda.1se
lasso_coeff <- lasso_model$glmnet.fit$beta[,lasso_model$glmnet.fit$lambda == best_lasso_lambda]
#lasso_coeff

##fit a ridge model

set.seed(123)
ridge_model <- cv.glmnet(as.matrix(con.train[,-(1:2)]), as.matrix(con.train[,2]),
                         lambda = 10^seq(4,-1,-1), alpha=0)
##ridge_model$lambda.1se
##plot(ridge_model)
best_ridge_lambda <- ridge_model$lambda.1se

ridge_coeff <- ridge_model$glmnet.fit$beta[,ridge_model$glmnet.fit$lambda == best_ridge_lambda]

##ridge_coeff


###fit an elasticnet model
set.seed(123)
elastic_model <- cv.glmnet(as.matrix(con.train[,-(1:2)]), as.matrix(con.train[,2]),
                           lambda = 10^seq(4,-1,-1), alpha=0.5)
##elastic_model$lambda.1se
##plot(elastic_model)
best_elastic_lambda <- elastic_model$lambda.1se

elastic_coeff <- elastic_model$glmnet.fit$beta[,elastic_model$glmnet.fit$lambda == best_elastic_lambda]

##elastic_coeff



## compare coefficients
coef = data.table(lasso = lasso_coeff,
                  ridge = ridge_coeff,
                  elastic = elastic_coeff)
coef[,feature:= names(ridge_coeff)]
library(ggplot2)
to_plot <- melt(coef, id.vars = 'feature', variable.name = 'model', value.name = 'coefficient')
ggplot(to_plot, aes (x=feature, y=coefficient, fill=model)) + coord_flip() + geom_bar(stat = 'identity') +
  facet_wrap(~ model) + guides(fill=FALSE)

##ridge predictions
results <-predict(ridge_model, s=best_ridge_lambda, as.matrix(con.valid[,-(1:2)]), type="response")
ridge_con <- confusionMatrix(as.factor(con.valid[,2]), as.factor(ifelse(results>0.5, 1, 0)))


##lasso predictions
lasso_results <-predict(lasso_model, s=best_lasso_lambda, as.matrix(con.valid[,-(1:2)]), type="response")
lasso_con <- confusionMatrix(as.factor(con.valid[,2]), as.factor(ifelse(lasso_results>0.5, 1, 0)))

##elasticnet predictions
elastic_results <-predict(elastic_model, s=best_elastic_lambda, as.matrix(con.valid[,-(1:2)]), type="response")
elasticnet_con <- confusionMatrix(as.factor(con.valid[,2]), as.factor(ifelse(elastic_results>0.5, 1, 0)))

options(scipen = 999)
accuracy <- vector()
accuracy[1] <- lasso_con$overall["Accuracy"]
accuracy[2] <- ridge_con$overall["Accuracy"]
accuracy[3] <- elasticnet_con$overall["Accuracy"]


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
AUC <- fun.auc(ifelse(results>0.5, 1, 0), con.valid[,2])


## RMSE Calculation
predictions <- ifelse(results>0.5, 1, 0)
observations <- con.valid[,2]

RMSE = (sum((predictions - observations)^2)/nrow(con.valid))^0.5
RMSE
1-RMSE+unique(AUC[AUC])
accuracy
