rm(list = ls(all=T))
hbt=read.csv("CSE7302c_train.csv",header = T)
data.class(hbt)
colnames(hbt)
str(hbt)
cols=c( "jet1b.tag" , "jet2b.tag" , "jet3b.tag" , "jet4b.tag" , "class" )
for (i in cols){
  hbt[,i]=as.factor(hbt[,i])
}
str(hbt)
hbt$id=NULL
head(hbt)

summary(hbt)

numeric_Variables = hbt[,-c(1,10,14,18,22)]
cat_cols=setdiff(colnames(hbt),numeric_Variables) 
data.class(numeric_Variables)
data.class(cat_cols)


##############

library(caret)
# 
hbt_predict_object <- preProcess(hbt[, -c(1,10,14,18,22)], method=c("center", "scale"))
hbt_newData <- predict(hbt_predict_object, hbt[, -c(1,10,14,18,22)])
head(hbt_newData)
data.class(hbt_newData)

#########

hbt_catcols=hbt[c(1,10,14,18,22)]

library(dplyr)
hbt_newdata_merged=cbind(hbt_newData,hbt_catcols)

##############


library(mlr)
hbt_dummy=createDummyFeatures(obj = hbt_newdata_merged,target = "class",method = "reference")
colnames(hbt_dummy)
head(hbt_dummy) 


set.seed(777)
hbt_ind <- sample(2, nrow(hbt_dummy), replace = TRUE, prob = c(0.7, 0.3))
hbt_train <- hbt_dummy[hbt_ind==1,]
hbt_test <- hbt_dummy[hbt_ind==2,]

##########

str(hbt_dummy)

hbt_data_logreg=hbt_dummy[,-c(2,3,5,7,8,11,10,13,17,27,30,34,38)]

##########

table(hbt_train$class)

hbt_x_train <- as.matrix(hbt_train[, -25])

hbt_y_train <- as.matrix(hbt_train[, 25])

hbt_x_test <- as.matrix(hbt_test[, -25])

hbt_y_test <- as.matrix(hbt_test[, 25])

library(glmnet)

hbt_cv=cv.glmnet(x = hbt_x_train,y=hbt_y_train,family="binomial",type.measure = "class",alpha=1,nlambda=100)

plot(hbt_cv)

fit_lasso=glmnet(x = hbt_x_train,y = hbt_y_train,family = "binomial",alpha = 1,lambda = hbt_cv$lambda.1se)
coef(fit_lasso)

plot(hbt_cv$glmnet.fit, xvar="lambda", label=TRUE)

hbt_lasso_variables=as.matrix(coef(hbt_cv, hbt_cv$lambda.1se))
hbt_lasso_variables
length(hbt_lasso_variables)
hbt_lasso_variables=data.frame(hbt_lasso_variables)
data.class(hbt_lasso_variables)
colnames(hbt_lasso_variables)=c("CoEff")
hbt_lasso_variables

hbt_lasso_variables[hbt_lasso_variables==0]=NA
(hbt_lasso_variables)
hbt_lasso_variables1=na.omit(hbt_lasso_variables)

###################################################

#Logistic Regression

log_ind <- sample(2, nrow(hbt_data_logreg), replace = TRUE, prob = c(0.7, 0.3))
log_train <- hbt_data_logreg[log_ind==1,]
log_test <- hbt_data_logreg[log_ind==2,]

colnames(log_train)
logistic=glm(class~.,data = log_train,family = "binomial")
summary(logistic)

pred=predict(logistic,newdata = log_train,type = "response")

table(log_train$class,pred>0.5)
(11732+18934)/(11732+18934+6423+10818)

library(ROCR)
pred_ROCR <- ROCR::prediction(pred,log_train$class)
perf <- ROCR::performance(pred_ROCR, "tpr","fpr")
plot(perf, col=rainbow(10),colorize = T,print.cutoffs.at=seq(0,1,0.05))
perf_auc <- ROCR::performance(pred_ROCR, measure = "auc")
auc <- perf_auc@y.values[[1]]
print(auc)


pred2=predict(logistic,newdata = log_test,type = "response")

table(log_test$class,pred2>0.5)
(5167+8118)/(5167+8118+2804+4640)

#####Test on test data#######
hbt_test_val=read.csv("CSE7302c_test.csv",header = T)
str(hbt_test_val)
for (i in cols){
  hbt_test_val[,i]=as.factor(hbt_test_val[,i])
}
hbt_test_val$jet4phi = as.numeric(as.character(hbt_test_val$jet4phi))
hbt_test_val$m_jj = as.numeric(as.character(hbt_test_val$m_jj))
hbt_test_val$m_jjj = as.numeric(as.character(hbt_test_val$m_jjj))
hbt_test_val$m_lv = as.numeric(as.character(hbt_test_val$m_lv))
hbt_test_val$m_jlv = as.numeric(as.character(hbt_test_val$m_jlv))
hbt_test_val$m_bb = as.numeric(as.character(hbt_test_val$m_bb))
hbt_test_val$m_wbb = as.numeric(as.character(hbt_test_val$m_wbb))
hbt_test_val$m_wwbb = as.numeric(as.character(hbt_test_val$m_wwbb))
hbt_test_val$id=NULL
str(hbt_test_val)
sum(is.na(hbt_test_val))
summary(hbt_test_val)
hbt_test_data <- hbt_test_val[!(hbt_test_val$jet4b.tag=="?"),] 
summary(hbt_test_data)
hbt_test_newData <- predict(hbt_predict_object, hbt_test_data[, -c(1,10,14,18,22)])
sum(is.na(hbt_test_newData))
hbt_test_catcols=hbt_test_data[c(1,10,14,18,22)]
hbt_test_newdata_merged=cbind(hbt_test_newData,hbt_test_catcols)
library(mlr)
hbt_test_dummy=createDummyFeatures(obj = hbt_test_newdata_merged,target = "class",method = "reference")
colnames(hbt_test_dummy) 
names(hbt_test_dummy)[names(hbt_test_dummy) == "jet4b.tag.1.550980687"] <- "jet4b.tag.1.550981"
names(hbt_test_dummy)[names(hbt_test_dummy) == "jet4b.tag.3.101961374"] <- "jet4b.tag.3.101961"
hbt_test_dummy$jet4b.tag.3.101961374 = 0
hbt_test_data_logreg=hbt_test_dummy[,-c(2,3,5,7,8,11,10,13,17,27,30,34,38)]
colnames(hbt_test_data_logreg)
pred_test=predict(logistic,newdata = hbt_test_data_logreg,type = "response")
#confusionMatrix(as.factor(pred_test), hbt_test_data_logreg$class)
table(hbt_test_data_logreg$class,pred_test>0.5)

val = ifelse(pred_test > 0.5, 1, 0)

(7405+11468)/(7405+11468+6461+4079)

pred_test_ROCR <- ROCR::prediction(pred_test,hbt_test_data_logreg$class)
perf_test <- ROCR::performance(pred_test_ROCR, "tpr","fpr")
head(pred_test_ROCR)
plot()

pred_test_ROCR

write.csv(data.frame(val), "file.csv")
