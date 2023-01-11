
####***********************************************************************####
####*                        Regression Modeling
####*                        Ambika Huluse Kapanaiah(u3227622)
####*                        Final Assignment
####*                        01/11/2021
####***********************************************************************####





#####*******************************************************************
##### Loading all the required packages
#####*******************************************************************

library(data.table)
library(ggplot2)
library(purrr)
library(tidyr)
library(dplyr)
library(GGally)
library(rmarkdown)
library(caret)
library(reshape2)
library(car)
library(modelr)

#####*******************************************************************
##### Read and prepare the data
#####*******************************************************************

#setwd("D:\\DataScience\\SEM2\\Reg_modelling\\Final_assignment")

Seoul_BikeSharing_df <- read.csv("SeoulBikeData.csv")
head(Seoul_BikeSharing_df)
View(Seoul_BikeSharing_df)
nrow(Seoul_BikeSharing_df)
ncol(Seoul_BikeSharing_df)
colnames(Seoul_BikeSharing_df)

###checking for Missing values
colSums(is.na(Seoul_BikeSharing_df))

### Removing the Date as it is not important
drops <- c("Date")
Seoul_BikeSharing_df <- Seoul_BikeSharing_df[ , !(names(Seoul_BikeSharing_df) %in% drops)]

#Seoul_BikeSharing_df$Hour <- as.factor(Seoul_BikeSharing_df$Hour)


colnames(Seoul_BikeSharing_df)

#####*******************************************************************
##### Renaming the column names for the ease of use.
##### Espesially the target name changed from Rented.Bike.Count to Demand
#####*******************************************************************


Seoul_BikeSharing_df <- Seoul_BikeSharing_df %>% 
  rename( Demand = Rented.Bike.Count, Temp_degc = Temperature..C.,
          Humidity = Humidity..., Wind_Speed = Wind.speed..m.s.,
          Visibility_10m = Visibility..10m.,
          Dewpt_temp_Degc = Dew.point.temperature..C.,
          Sol_Rad_MJm2 = Solar.Radiation..MJ.m2.,
          Rainfall_mm = Rainfall.mm., Snowfall_cm = Snowfall..cm.)

colnames(Seoul_BikeSharing_df)


#####*******************************************************************
##### EDA of Numerical columns
#####*******************************************************************


#All Numeric cols dataframe creation
BikeSharing_NumericCols <- Seoul_BikeSharing_df %>%
  keep(is.numeric)
colnames(BikeSharing_NumericCols)

# histogram of all numeric cols
DataFrame_hist <- melt( BikeSharing_NumericCols )
ggplot(data = DataFrame_hist , aes(value, fill = variable)) + 
  geom_histogram(bins = 10,colour= "blue") + 
  ylab("Number of observations") +
  facet_wrap(~ variable, scales = "free_x")

##Scatter plot of all the numerical columns
DataFrame_scatter <- melt(BikeSharing_NumericCols, "Demand")


ggplot(DataFrame_scatter, aes(value, Demand)) + 
  geom_point() + 
  facet_wrap(~variable, scales = "free")

ggcorr(BikeSharing_NumericCols, label = TRUE)


#####*******************************************************************
##### EDA of Categorical features
#####*******************************************************************

## Creating categorical columns data frame
BikeSharing_CategoricCols <- Seoul_BikeSharing_df %>% 
  select_if(negate(is.numeric))
colnames(BikeSharing_CategoricCols)

#Adding target feature to categorical dataframe to visualize dependency
BikeSharing_CategoricCols$Demand   <- Seoul_BikeSharing_df$Demand


##Hour versus Demand
ggplot(BikeSharing_CategoricCols,aes(Hour,Demand, fill=factor(Hour)))+
  geom_boxplot()

##Seasons versus Demand
ggplot(BikeSharing_CategoricCols,aes(Seasons,Demand, fill=factor(Seasons)))+
  geom_boxplot()

##Holiday versus Demand
ggplot(BikeSharing_CategoricCols,aes(Holiday,Demand, fill=factor(Holiday)))+
  geom_boxplot()

##Functioning.Day versus Demand
ggplot(BikeSharing_CategoricCols,aes(Functioning.Day,Demand, fill=factor(Functioning.Day)))+
  geom_boxplot()

#####*******************************************************************
##### Checking for Outliers in the target feature.
#####*******************************************************************

out <- boxplot.stats(Seoul_BikeSharing_df$Demand)$out
out
ggplot(Seoul_BikeSharing_df) +
  aes(x = "", y = Demand) +
  geom_boxplot(fill = "#0c4c8a") +
  theme_minimal()


#####*******************************************************************
##### Creating dummy variables of all categorical columns
#####*******************************************************************

Seoul_BikeSharing_df %>% select_if(negate(is.numeric)) %>%
  data.frame()

# Install the required package
#install.packages("fastDummies")
# Load the library
library(fastDummies)
# Using PlantGrowth dataset
data_df <- Seoul_BikeSharing_df
# Create dummy variable
data_df <- dummy_cols(data_df, 
                      select_columns = c("Seasons", "Hour"))

colnames(data_df)

data_df$Holiday <- ifelse(data_df$Holiday == 'Holiday', 1,0)

data_df$Functioning.Day <- ifelse(data_df$Functioning.Day == 'No', 0,1)



#removing Dewpt also as it is under muticollinearity problem
drops <- c("Hour", "Seasons","Seasons_Winter", "Hour_23","Dewpt_temp_Degc")
Seoul_BikeSharingdummy <- data_df[ , !(names(data_df) %in% drops)]
Seoul_BikeSharing_Poisson <- Seoul_BikeSharingdummy
Seoul_BikeSharing_Poisson$Dewpt_temp_Degc <-  Seoul_BikeSharing_df$Dewpt_temp_Degc


#dataset without outlier treated and includes all predictors.
Seoul_BikeSharingModel1 <- Seoul_BikeSharingdummy
Seoul_BikeSharingModel1$Dewpt_temp_Degc <-  data_df$Dewpt_temp_Degc
head(Seoul_BikeSharingModel1)



#####*******************************************************************
##### Creating data frames removing outlier rows.
#####*******************************************************************
#Removing outlier rows with the quantiles of 
# Lower bound 0.2 and upperbound 0.80 (Based on Outliers in target feature)

lower_bound <- quantile(Seoul_BikeSharingdummy$Demand, 0.20)
upper_bound <- quantile(Seoul_BikeSharingdummy$Demand, 0.80)

outlier_ind <- which(Seoul_BikeSharingdummy$Demand < lower_bound | Seoul_BikeSharingdummy$Demand > upper_bound)

Seoul_BikeSharingModel2 <- Seoul_BikeSharingdummy[-outlier_ind, ]

####********************************************************************
####* Taking squareroot transformation for response
#####*******************************************************************

Seoul_BikeSharingModel1$Demand_sqrt <- sqrt(Seoul_BikeSharingModel1$Demand)
Seoul_BikeSharingModel2$Demand_sqrt <- sqrt(Seoul_BikeSharingModel2$Demand)

head(Seoul_BikeSharingModel1)
head(Seoul_BikeSharingModel2)


################Multiple Linear Regression#############################

# dropping demand for both model datasets as we have squareroot of demand.
drop = c("Demand")
Seoul_BikeSharingMLM1 <- Seoul_BikeSharingModel1[ , !(names(Seoul_BikeSharingModel1) %in% drop)]
Seoul_BikeSharingMLM2 <- Seoul_BikeSharingModel2[ , !(names(Seoul_BikeSharingModel2) %in% drop)]
head(Seoul_BikeSharingMLM1)
head(Seoul_BikeSharingMLM2)


#####*******************************************************************
##### Splitting dataset
#####*******************************************************************


#**************fittiing with Seoul_BikeSharingMLM1
set.seed(123) 
split <- resample_partition(Seoul_BikeSharingMLM1, c(train=0.8, test = 0.2))

Seoul_BikeSharingMLM1_train <- data.frame(split$train)
Seoul_BikeSharingMLM1_test <- data.frame(split$test)


MLM_1 <- lm(Demand_sqrt ~., 
    data = Seoul_BikeSharingMLM1)


summary(MLM_1)



MLM1predictions <- predict(MLM_1, Seoul_BikeSharingMLM1_test)

plot(MLM_1,1)
plot(MLM_1,2)
plot(MLM_1,3)
plot(MLM_1,4)



#######


#**************fittiing with Seoul_BikeSharingMLM2

#Without outliers
set.seed(123) 
split <- resample_partition(Seoul_BikeSharingMLM2, c(train=0.8, test = 0.2))

Seoul_BikeSharingMLM2_train <- data.frame(split$train)
Seoul_BikeSharingMLM2_test <- data.frame(split$test)



MLM_2 <-  lm(Demand_sqrt ~ ., 
             data = Seoul_BikeSharingMLM2)


MLM_2predictions <- predict(MLM_2, Seoul_BikeSharingMLM2_test)

summary(MLM_2)
plot(MLM_2,1)
plot(MLM_2,2)
plot(MLM_2,3)
plot(MLM_2,4)


Poissionsummary1 <- data.frame(RMSE = RMSE(MLM_2predictions, Seoul_BikeSharingMLM2_test$Demand_sqrt),
                               RSQ = caret::R2(MLM_2predictions, Seoul_BikeSharingMLM2_test$Demand_sqrt),
                               MAE = MAE(MLM_2predictions, Seoul_BikeSharingMLM2_test$Demand_sqrt))
Poissionsummary1


#############End of Multiple Linear Regression#############################





#######################Elastic-net Regression##################################


drop = c("Demand_sqrt")
Seoul_BikeSharingEL1 <- Seoul_BikeSharingModel1[ , !(names(Seoul_BikeSharingModel1) %in% drop)]


set.seed(123) 
split <- resample_partition(Seoul_BikeSharingModel1, c(train=0.8, test = 0.2))

Seoul_BikeSharingEl1_train <- data.frame(split$train)
Seoul_BikeSharingEl1_test <- data.frame(split$test)


elastic1 <- train(
  Demand~., data=Seoul_BikeSharingEL1, method="glmnet", 
  trControl=trainControl("cv", number=10)
)

coef(elastic1$finalModel, elastic1$bestTune$lambda)
predictions <- elastic1 %>% predict(Seoul_BikeSharingEl1_test)

data.frame(
  RMSE.net = RMSE(predictions, Seoul_BikeSharingEl1_test$Demand),
  Rsquare.net = caret::R2(predictions, Seoul_BikeSharingEl1_test$Demand)
)



Seoul_BikeSharingEL2 <- Seoul_BikeSharingMLM1

set.seed(123) 
split <- resample_partition(Seoul_BikeSharingEL2, c(train=0.8, test = 0.2))

Seoul_BikeSharingEL2_train <- data.frame(split$train)
Seoul_BikeSharingEl2_test <- data.frame(split$test)


elastic2 <- train(
  Demand_sqrt~., data=Seoul_BikeSharingEL2, method="glmnet", 
  trControl=trainControl("cv", number=10)
)

coef(elastic2$finalModel, elastic2$bestTune$lambda)
predictions <- elastic2 %>% predict(Seoul_BikeSharingEl2_test)

data.frame(
  RMSE.net = RMSE(predictions, Seoul_BikeSharingEl2_test$Demand_sqrt),
  Rsquare.net = caret::R2(predictions, Seoul_BikeSharingEl2_test$Demand_sqrt)
)


########################KNN -Regression####################################

Seoul_BikeSharingKNN1 <- Seoul_BikeSharingModel1

set.seed(123) 
split <- resample_partition(Seoul_BikeSharingKNN1, c(train=0.8, test = 0.2))

Seoul_BikeSharingKNN1_train <- data.frame(split$train)
Seoul_BikeSharingKNN1_test <- data.frame(split$test)


knn_1 <- train(Demand_sqrt ~ ., 
             data = Seoul_BikeSharingKNN1,
             method = "knn",
             trControl = trainControl("cv", number = 10),
             preProcess=c("center", "scale"),
             #"center" subtracts the mean of the predictor's data 
             #(again from the data in x) from the predictor values 
             #while "scale" divides by the standard deviation
             tuneLength = 20
)

knn_1
plot(knn_1)


knnpredictions1 <- predict(knn_1, Seoul_BikeSharingKNN1_test)

knnsummary1 <- data.frame(RMSE = RMSE(knnpredictions1, Seoul_BikeSharingKNN1_test$Demand_sqrt),
                         RSQ = caret::R2(knnpredictions1, Seoul_BikeSharingKNN1_test$Demand_sqrt),
                         MAE = MAE(knnpredictions1, Seoul_BikeSharingKNN1_test$Demand_sqrt))
knnsummary1




##########################Poisson model####################################


Seoul_BikeSharing_Poisson
hist(Seoul_BikeSharing_Poisson$Demand)
attach(Seoul_BikeSharing_Poisson)


myPoissonmodel <- glm(Demand ~., family=poisson(link="log"),
                      data = Seoul_BikeSharing_Poisson)
summary(myPoissonmodel)


par(mfrow = c(2,1))

plot(myPoissonmodel, which = c(1,2))
plot(myPoissonmodel, which = c(3,4))
  

set.seed(123) 
split <- resample_partition(Seoul_BikeSharing_Poisson, c(train=0.8, test = 0.2))

Seoul_BikeSharing_Poisson1_train <- data.frame(split$train)
Seoul_BikeSharing_Poisson1_test <- data.frame(split$test)


Poissonprediction1 <- predict(myPoissonmodel, Seoul_BikeSharing_Poisson1_test)

PoissonSummary1 <- data.frame(RMSE = RMSE(Poissonprediction1, Seoul_BikeSharing_Poisson1_test$Demand),
                          RSQ = caret::R2(Poissonprediction1, Seoul_BikeSharing_Poisson1_test$Demand),
                          MAE = MAE(Poissonprediction1, Seoul_BikeSharing_Poisson1_test$Demand))
PoissonSummary1


####################---------End of the code--------#############################
