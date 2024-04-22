#load the required library
library(class)

#import the dataset
data <- read.csv("fidelity.csv")

#set the seed for reproducibility
set.seed(222)

#prepare data set for analysis (remove id and x and last name and first name)
data <- data[, -c(1:3)]
head(data)

#transform the test into numeric ("Did not subscribed = 0" and " "Subscribed = 1)
data$decision_num <- ifelse(data$decision == "Subscribed", 1, 0)
#remove the column 
data<-data[,-5]


#split the data into training and testing sets
#training set is 375
train_indices <- sample(1:nrow(data), 375)
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

#extract predictor [dependent] variables (age, amount, visits) and target [independent] variable (decision)
predictors <- train_data[, c("age", "amount", "visits")]
target <- train_data$decision_num

#normalize the predictor variables
predictors <- scale(predictors)

#fit the KNN model, k is chosen to be 25
k <- 25
knn_model <- knn(train = predictors, test = scale(test_data[, c("age", "amount", "visits")]), cl = target, k = k)

#print the predictions
print(knn_model)


# Question: how good is this model?
#predict the target variable for the test data using the trained KNN model
predicted <- knn(train = scale(train_data[, c("age", "amount", "visits")]), 
                 test = scale(test_data[, c("age", "amount", "visits")]), 
                 cl = train_data$decision_num, 
                 k = k)

#calculate accuracy
accuracy <- mean(predicted == test_data$decision_num)
cat("Accuracy:", accuracy)

#calculate confusion matrix
confusion_matrix <- table(predicted, test_data$decision_num)
cat("Confusion Matrix: ", confusion_matrix)

#calculate precision, recall, and F1-score
precision <- confusion_matrix[2,2] / sum(confusion_matrix[,2])
recall <- confusion_matrix[2,2] / sum(confusion_matrix[2,])
f1_score <- 2*(precision * recall) / (precision + recall)

cat("Precision:", precision)
cat("Recall:", recall)
cat("F1-Score:", f1_score)


#results: true positives = 41 (accurate), false positives = 9, true negatives =54 (accurate), and false negatives= 24.
#32% of the customers predicted to subscribe to the loyalty program actually did so.
#model correctly identified about 73% of the customers who actually subscribed to the loyalty program.
#Overall, the model achieved a relatively high recall, indicating its ability to correctly identify customers who subscribed to the loyalty program, 
#but has a lower precision and F1-score. This suggests that the model may have a higher rate of false positives, meaning it incorrectly predicts some customers to subscribe when they do not. 
#Further optimization or fine-tuning of the model may be necessary to improve its performance.