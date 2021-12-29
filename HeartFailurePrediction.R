# loading the libraries required for this code
library("e1071")
library(tidyverse)
library(dummies)
library(corrplot)
library(olsrr)
library(smotefamily)
library(rpart)
library(rpart.plot)
library(class)
library(neuralnet)

# setting the working directory to the desired folder
setwd("C:/Users/UAL-Laptop/OneDrive - University of Arizona/Documents/MIS545 Project")

# 1. Logistic Regression algorithm

# reading the csv in a tibble 
heartAttack <- read_csv(file = "heart.csv",
                        col_types = "iliiiliilniiil",
                        col_names = TRUE)

# print, see the structure and get the summary of the tibble
print(heartAttack)
str(heartAttack)
summary(heartAttack)

# define a function to display all histograms
displayAllHistograms <- function(tibbleDataSet) {
  tibbleDataSet %>%
    keep(is.numeric) %>%
    gather() %>%
    ggplot() + geom_histogram(mapping = 
                                aes(x=value, fill=key),
                              color = "black") + 
    facet_wrap(~key, scales = "free") + theme_minimal()
}

# call the function to get all the histograms
displayAllHistograms(heartAttack)

# display the correlation matix
round(cor(heartAttack), 2)

# display the correlation plot
corrplot(cor(heartAttack), method = "number", 
         type="lower")

# remove cols which has value more than 0.7
heartAttack2 <- select(.data=heartAttack,
                       age,
                       sex,
                       cp,
                       trtbps,
                       chol,
                       fbs,
                       restecg,
                       thalachh,
                       exng,
                       oldpeak,
                       slp,
                       caa,
                       thall,
                       output)

# Split the data into traning and test
set.seed(203)

# sample set of 75%
sampleSet <- sample(nrow(heartAttack2),
                    round(nrow(heartAttack2) * 0.75), replace = FALSE)

# training data set
heartAttackTraining <- heartAttack2[sampleSet, ]

# test data set
heartAttackTesting <- heartAttack2[-sampleSet, ]

# Check the class imblanance issue in the training dataset
summary(heartAttackTraining$output)

# check the magnitude
cancelledService <- 109 / 118

# deal with the class imbalance using the SMOTE technique
heartAttackTrainingSmoted <- tibble(SMOTE(X=data.frame(heartAttackTraining),
                                          target = heartAttackTraining$output,
                                          dup_size = 3) $ data)

# check the summary
summary(heartAttackTrainingSmoted)

# convert the cancelled service back to logical
heartAttackTrainingSmoted <- heartAttackTrainingSmoted %>%
  mutate (output = as.logical(output),
          sex = as.logical(sex),
          fbs = as.logical(fbs),
          exng = as.logical(exng))

# remove the extra class added
heartAttackTrainingSmoted <- heartAttackTrainingSmoted %>% 
  select(-class)

# check the class imbalance in smoted dataset
summary(heartAttackTrainingSmoted)

# generate the logistic regression model
heartAttackModel <- glm(data = heartAttackTrainingSmoted,
                        family = binomial,
                        formula = output ~ .)

# get the summary of the model
summary(heartAttackModel)

# generate the odds ration for all the independent variables
exp(coef(heartAttackModel))["age"]
exp(coef(heartAttackModel))["sex"]
exp(coef(heartAttackModel))["cp"]
exp(coef(heartAttackModel))["trbps"]
exp(coef(heartAttackModel))["chol"]
exp(coef(heartAttackModel))["fbs"]
exp(coef(heartAttackModel))["restecg"]
exp(coef(heartAttackModel))["thalachh"]
exp(coef(heartAttackModel))["exng"]
exp(coef(heartAttackModel))["oldpeak"]
exp(coef(heartAttackModel))["slp"]
exp(coef(heartAttackModel))["caa"]
exp(coef(heartAttackModel))["thall"]
exp(coef(heartAttackModel))["output"]

# predict outcomes in testing datasets
heartAttackPrediction <- predict(heartAttackModel,
                                 heartAttackTesting,
                                 type= "response")

# display on console
print(heartAttackPrediction)

# dividing in 0 and 1
heartAttackPrediction <- ifelse(heartAttackPrediction >= 0.5, 1, 0)

# display on console
print(heartAttackPrediction)

# generate the confusion matrix
heartAttackConfusionMatrix <- table(heartAttackTesting$output,
                                    heartAttackPrediction)

# display on console
print(heartAttackConfusionMatrix)

# calculating the false positive rate
print(heartAttackConfusionMatrix[1,2] / 
        heartAttackConfusionMatrix[1,2] + heartAttackConfusionMatrix[1,1])

# calculate the false negative rate
print(heartAttackConfusionMatrix[2,1] / 
        heartAttackConfusionMatrix[2,1] + heartAttackConfusionMatrix[2,2])

# calculating the predictive accuracy
sum(diag(heartAttackConfusionMatrix)) / nrow(heartAttackTesting)

#-------------------------------------------------------------------------------

# 2. Decision Tree Alogrithm

# read csv into tibble
heartAttack <- read_csv(file = "heart.csv",
                        col_types = "iliiiliilniiil",
                        col_names = TRUE)

# display the heartAttack tibble on the console
print(heartAttack)

# display the structure of the heartAttack tibble
summary(heartAttack)

# split data into training and testing
# using the set.seed() function to ensure we can get the same result
# everytime we run a random sampling process
set.seed(370)

# create a vector of 75% randomly sampled rows from original dataset
sampleSet <- sample(nrow(heartAttack),
                    round(nrow(heartAttack) * 0.75),
                    replace = FALSE)

# put the records from 75% sample into Training
heartAttackTraining <- heartAttack[sampleSet, ]

# rest of the data in Testing (25% of records)
heartAttackTesting <- heartAttack[-sampleSet, ]

# Train the decision tree model using the training data set. Note 
# the complexity parameter of 0.01 is the default value
heartAttackDecisionTreeModel <- rpart(formula = output ~.,
                                      method = "class",
                                      cp = 0.01,
                                      data = heartAttackTraining)

# display the decision tree plot
rpart.plot(heartAttackDecisionTreeModel)

# predict classes for each record in the testing dataset
heartAttackPrediction <- predict(heartAttackDecisionTreeModel,
                                 heartAttackTesting,
                                 type = "class")

# display the predictions from heartAttackPrediction on the console
print(heartAttackPrediction)

# evaluate the model by forming a confusion matrix
heartAttackConfusionMatrix <- table(heartAttackTesting$output,
                                    heartAttackPrediction)

# display the confusion matrix on the console
print(heartAttackConfusionMatrix)

# calculate the model predictive accuracy
predictiveAccuracy <- sum(diag(heartAttackConfusionMatrix)) / 
  nrow(heartAttackTesting)

# display the predictive accuracy on the console
print (predictiveAccuracy)

# generate the model with different complexity parameter 
# train the decision tree model using the training data set
heartAttackDecisionTreeModel <- rpart(formula = output ~.,
                                      method = "class",
                                      cp = 0.007,
                                      data = heartAttackTraining)

# display the decision tree plot 
rpart.plot(heartAttackDecisionTreeModel)

# evaluate the model by forming a confusion matrix
heartAttackConfusionMatrix <- table(heartAttackTesting$output,
                                    heartAttackPrediction)

# display the confusion matrix on the console
print(heartAttackConfusionMatrix)

# calculate the model predictive accuracy
predictiveAccuracy <- sum(diag(heartAttackConfusionMatrix)) / 
  nrow(heartAttackTesting)

# display the predictive accuracy on the console
print (predictiveAccuracy)

# ------------------------------------------------------------------------------

# 3. K - Nearest Neighbor Algorithm

# read csv into tibble
heartAttack <- read_csv(file = "heart.csv",
                        col_types = "iliiiliilniiil",
                        col_names = TRUE)

# display the heartAttack tibble on the console
print(heartAttack)

# display summay of the heartAttack tibble
summary(heartAttack)

# separate the tibble into 2. One with just the label and
# one with other variables
heartAttackLabels <- heartAttack %>% select(output)
heartAttack <- heartAttack %>% select(-output)

# creating a function DisplayAllHistograms that takes tibble parameters
displayAllHistograms <- function(tibbleDataset) {
  tibbleDataset %>% 
    keep(is.numeric) %>%
    gather() %>%
    ggplot() + geom_histogram(mapping = aes(x=value, fill=key),
                              color = "black") +
    facet_wrap (~key, scales = "free") +
    theme_minimal ()
}

# calling the displayAllHistogram() function, with tibble as the parameter  
displayAllHistograms(heartAttack)


# split data into training and testing
# using the set.seed() function to ensure we can get the same result
# everytime we run a random sampling process
set.seed(517)

# create a vector of 75% randomly sampled rows from original dataset
sampleSet <- sample(nrow(heartAttack),
                    round(nrow(heartAttack) * 0.75),
                    replace = FALSE)

# put the records from 75% sample into heartAttackTraining
heartAttackTraining <- heartAttack[sampleSet, ]
heartAttackTrainingLabels <- heartAttackLabels[sampleSet, ]

# rest of the data in heartAttackTesting (25% of records)
heartAttackTesting <- heartAttack[-sampleSet, ]
heartAttackTestingLabels <- heartAttackLabels[-sampleSet, ]

# generate the k-Nearest Neighbors Model
heartAttackPrediction <- knn(train = heartAttackTraining,
                             test = heartAttackTesting, 
                             cl = heartAttackTrainingLabels$output,
                             k = 7)

# display predictions from the testing dataset on the console
print(heartAttackPrediction)

# display summary of the predictions from testing dataset
print(summary(heartAttackPrediction))

# Evaluating the model by forming a confusion matrix
heartAttackConfusionMatrix <- table(heartAttackTestingLabels$output,
                                    heartAttackPrediction)

# Displaying the confusion matrix on the console
print(heartAttackConfusionMatrix)

# Calculating the model predictive accuracy
predictiveAccuracy <- sum(diag(heartAttackConfusionMatrix)) /
  nrow(heartAttackTesting)
print(predictiveAccuracy)

# Create a matrix of k-values with their predictive accuracy
kValueMatrix <- matrix(data = NA,
                       nrow = 0,
                       ncol = 2)
# assigning column names of "k value" and "Predictive accuracy" to the 
# kValueMatrix
colnames(kValueMatrix) <- c("k value", "Predictive Accuracy")

# Loop through odd values of k from 1 up to the number of records 
# in the training dataset. With each pass through the loop, store the k-value 
# along with its predictive accuracy.
for (kValue in 1:nrow(heartAttackTraining)) {
  # only calculate predictive value if the k value is odd
  if(kValue %% 2 !=0) {
    # generate the model
    heartAttackPrediction <- knn(train = heartAttackTraining,
                                 test = heartAttackTesting, 
                                 cl = heartAttackTrainingLabels$output,
                                 k = kValue)
    # generate the confusion matrix
    heartAttackConfusionMatrix <- table(heartAttackTestingLabels$output,
                                        heartAttackPrediction)
    # calculate the predictive accuracy
    predictiveAccuracy <- sum(diag(heartAttackConfusionMatrix)) /
      nrow(heartAttackTesting)
    
    # add a new row to the kValueMatrix
    kValueMatrix <- rbind(kValueMatrix, c(kValue, predictiveAccuracy))
  }
}

print(kValueMatrix)

# ------------------------------------------------------------------------------

# 4. Naive Bayes Algorithm

# read csv into tibble
heartAttack <- read_csv(file = "heart.csv",
                        col_types = "iliiiliilniiil",
                        col_names = TRUE)

# display the heartAttack tibble on the console
print(heartAttack)

# display the structure of heartAttack tibble
str(heartAttack)

# display summay of the heartAttack tibble
summary(heartAttack)

# split data into training and testing
# using the set.seed() function to ensure we can get the same result
# everytime we run a random sampling process
set.seed(154)

# create a vector of 75% randomly sampled rows from original dataset
sampleSet <- sample(nrow(heartAttack),
                    round(nrow(heartAttack) * 0.75),
                    replace = FALSE)

# put the records from 75% sample into training
heartAttackTraining <- heartAttack[sampleSet, ]

# rest of the data in testing (25% of records)
heartAttackTesting <- heartAttack[-sampleSet, ]

# Train the naive bayes model
heartAttackModel <- naiveBayes(formula = output ~.,
                               data = heartAttackTraining,
                               laplace = 1)

# build probabilities for each record in the testing dataset
heartAttackProbability <- predict(heartAttackModel,
                                  heartAttackTesting,
                                  type = "raw")

# display the probabilities from heartAttackProbability on the console
print(heartAttackProbability)

# predict classes for each record in the testing dataset
heartAttackPrediction <- predict(heartAttackModel,
                                 heartAttackTesting,
                                 type = "class")

# display the predictions from heartAttackPrediction on the console
print(heartAttackPrediction)

# evalute the model by forming a confusion matrix 
heartAttackConfusionMatrix <- table(heartAttackTesting$output,
                                    heartAttackPrediction)

# display the confusion matrix on console
print(heartAttackConfusionMatrix)

# calculate the predictive accuracy of the model
predictiveAccuracy <- sum(diag(heartAttackConfusionMatrix)) /
  nrow(heartAttackTesting)

# display the predictive accuracy on the console
print(predictiveAccuracy)

# ------------------------------------------------------------------------------

# 5. Neural Networks Algorithm

# Read the csv file into a tibble called heartAttack
heartAttack <- read_csv(file = "heart.csv",
                        # Set attribute type
                        col_types = "iliiiliilniiil",
                        col_names = TRUE)
# Display heartAttack in the console
print(heartAttack)
# Display the structure of heartAttack in the console
str(heartAttack)
# Display the summary of heartAttack in the console
summary(heartAttack)

# scale the age deature from 0 to 1
heartAttack <- heartAttack %>%
  mutate(ageScaled = (age - min(age)) / 
           (max(age) - min(age)))

# scale the sex deature from 0 to 1
heartAttack <- heartAttack %>%
  mutate(sexScaled = (sex - min(sex)) / 
           (max(sex) - min(sex)))

# scale the ChestPain deature from 0 to 1
heartAttack <- heartAttack %>%
  mutate(cpScaled = (cp - min(cp)) / 
           (max(cp) - min(cp)))

# scale the trtbps deature from 0 to 1
heartAttack <- heartAttack %>%
  mutate(trtbpsScaled = (trtbps - min(trtbps)) / 
           (max(trtbps) - min(trtbps)))

# scale the chol deature from 0 to 1
heartAttack <- heartAttack %>%
  mutate(cholScaled = (chol - min(chol)) / 
           (max(chol) - min(chol)))

heartAttack <- heartAttack %>%
  mutate(fbsScaled = (fbs - min(fbs)) / 
           (max(fbs) - min(fbs)))

heartAttack <- heartAttack %>%
  mutate(restecgScaled = (restecg - min(restecg)) / 
           (max(restecg) - min(restecg)))

heartAttack <- heartAttack %>%
  mutate(thalachhScaled = (thalachh - min(thalachh)) / 
           (max(thalachh) - min(thalachh)))

heartAttack <- heartAttack %>%
  mutate(exngScaled = (exng - min(exng)) / 
           (max(exng) - min(exng)))

heartAttack <- heartAttack %>%
  mutate(oldpeakScaled = (oldpeak - min(oldpeak)) / 
           (max(oldpeak) - min(oldpeak)))

heartAttack <- heartAttack %>%
  mutate(slpScaled = (slp - min(slp)) / 
           (max(slp) - min(slp)))

heartAttack <- heartAttack %>%
  mutate(caaScaled = (caa - min(caa)) / 
           (max(caa) - min(caa)))

heartAttack <- heartAttack %>%
  mutate(thallScaled = (thall - min(thall)) / 
           (max(thall) - min(thall)))


# Randomly split the dataset into heartAttackTraining (75% of records) 
#and heartAttackTesting (25% of records) using 591 as the random seed
set.seed(591)
sampleSet <- sample(nrow(heartAttack),
                    round(nrow(heartAttack) * 0.75),
                    replace = FALSE)
heartAttackTraining <- heartAttack[sampleSet, ]
heartAttackTesting <- heartAttack [-sampleSet, ]
# Generate the neural network model to predict output (dependent variable) 
# using all independent variables. 
# Use 3 hidden layers. Use "logistic" as the smoothing method and set 
# linear.output to FALSE.
heartAttackNeuralNet <- neuralnet(
  formula = output ~ ageScaled + sexScaled + cpScaled + trtbpsScaled + cholScaled + fbsScaled +restecgScaled + thalachhScaled
  + exngScaled + oldpeakScaled + slpScaled + caaScaled + thallScaled,
  data = heartAttackTraining,
  hidden = 3,
  act.fct = "logistic",
  linear.output = FALSE
)
# Display the neural network numeric results
print(heartAttackNeuralNet$result.matrix)

# Visualize the neural network
plot(heartAttackNeuralNet)

# Use heartAttackNeuralNet to generate probabilities on the 
# heartAttackTesting data set and store this in heartAttackProbability
heartAttackProbability <- compute(heartAttackNeuralNet,
                                  heartAttackTesting)

# Display the probabilities from the testing dataset on the console
print(heartAttackProbability)

# Convert probability predictions into 0/1 predictions and store this into 
# heartAttackPrediction
heartAttackPrediction <-
  ifelse(heartAttackProbability$net.result > 0.5, 1, 0)

# Display the 0/1 predictions on the console
print(heartAttackPrediction)

# Evaluate the model by forming a confusion matrix
heartAttackConfusionMatrix <- table(heartAttackTesting$output,
                                    heartAttackPrediction)

# Display the confusion matrix on the console
print(heartAttackConfusionMatrix)

# Calculate the model predictive accuracy
predictiveAccuracy <- sum(diag(heartAttackConfusionMatrix)) /
  nrow(heartAttackTesting)

# Display the predictive accuracy on the console
print(predictiveAccuracy)

