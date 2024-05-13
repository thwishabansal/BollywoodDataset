#Load libraries
library(dplyr)
library(tidyverse)
library(glmnet)
library(Ecdat)
library(ggplot2)
library(GGally)
library(dplyr)
library(rpart)
library(rpart.plot)
library(Metrics)
library(caret) # Needed for many functions
library(car)

# Replace 'your_username' with your actual username
file_path <- "/Users/thwishaabansal/Desktop/bollywood_data.csv"

# Read the dataset into R
bollywood <- read.csv(file_path)
View(bollywood)

# View the first few rows of the dataset
bollywood$Release_Date <- as.factor(bollywood$Release_Date)
bollywood$Genre <- as.factor(bollywood$Genre)
bollywood$Movie_Content <- as.factor(bollywood$Movie_Content)
bollywood$Dir_CAT <- as.factor(bollywood$Dir_CAT)
bollywood$LEAD_ACTOR_CAT <- as.factor(bollywood$LEAD_ACTOR_CAT)
bollywood$Item_Song <- as.factor(bollywood$Item_Song)
bollywood$Production_House_CAT <- as.factor(bollywood$Production_House_CAT)
bollywood$Music_Dir_CAT <- as.factor(bollywood$Music_Dir_CAT)
bollywood$Budget.in.crores <- as.numeric(bollywood$Budget.in.crores)
bollywood$Success.Faliure <- as.numeric(bollywood$Success.Faliure)

# Check the structure of the dataframe
str(bollywood)

# Set seed and split data
set.seed(123) 
train_indices <- sample(1:nrow(bollywood), 0.8 * nrow(bollywood))
train_data <- bollywood[train_indices, ]
test_data <- bollywood[-train_indices, ]

#Rename the variables:
names(bollywood) <- c(
  "movie_name",
  "release_date_festive",
  "genre",
  "movie_content",
  "director_category",
  "lead_actor_category",
  "has_item_song",
  "production_house_category",
  "music_director_category",
  "box_office_collection",
  "profit",
  "earning_ratio",
  "success_failure",
  "budget",
  "youtube_views",
  "youtube_likes",
  "youtube_dislikes",
  "X", 
  "release_date",
  "genre",
  "has_item_song",
  "budget"
)

View(bollywood)

#Question1
#A
season_anova <- aov(box_office_collection ~ release_date_festive, data = bollywood)
summary(season_anova)
TukeyHSD(season_anova)

#B
itemsong_anova <- aov(box_office_collection ~ has_item_song, data = bollywood)
summary(itemsong_anova)

#C
mod_3 <- aov(bollywood$budget ~ bollywood$movie_content)
summary(mod_3)
TukeyHSD(mod_3)
gh <- data.frame(
  bollywood$box_office_collection,
  bollywood$youtube_dislikes,
  bollywood$youtube_likes,
  bollywood$youtube_views,
  bollywood$budget,
  bollywood$success_failure,
  bollywood$earning_ratio,
  bollywood$profit
)
ggcorr(gh)

# Rename the variables:
names(gh) <- c(
  "box_office_collection",
  "youtube_dislikes",
  "youtube_likes",
  "youtube_views",
  "budget",
  "success_failure",
  "earning_ratio",
  "profit"
)

#Regression model for digital medium
regr_mod <- lm(box_office_collection ~ youtube_dislikes + youtube_likes + youtube_views, data = bollywood)
summary(regr_mod)

#D 
lead_actor_anova <- aov(box_office_collection ~ lead_actor_category, data = bollywood)
summary(lead_actor_anova)
tukey_result <- TukeyHSD(lead_actor_anova)
tukey_result

#E
mod_anova <- aov(budget ~ movie_content, data = bollywood)
summary(mod_anova)
tukey_result <- TukeyHSD(mod_anova)
tukey_result

#Question 2
#A
logit_mod <- glm(success_failure ~ budget, data = bollywood, family = binomial)
summary(logit_mod)
predicted_probs <- predict(logit_mod, type = "response")
budget_for_equal_likelihood <- predict(logit_mod, newdata = data.frame(budget = mean(bollywood$budget)), type = "link")
probability_equal_likelihood <- plogis(budget_for_equal_likelihood)
cat("The budget for which box office success and failure are equally likely is approximately:", round(mean(bollywood$budget), 2), "crores.\n")

#C
# Predict probability of success for a movie with a 100-crore budget
budget_100_crore <- 100
prob_success_100_crore <- predict(logit_mod, newdata = data.frame(budget = budget_100_crore), type = "response")
print(paste("Probability of success for a movie with a 100-crore budget:", prob_success_100_crore))

#d
#D
predicted_classes <- ifelse(predict(logit_mod, newdata = test_data, type = "response") > 0.5, 1, 0)
predicted_classes = factor(predicted_classes)
confusionMatrix(predicted_classes, test_data$Success_Faliure, positive = "1", mode = "prec_recall")

conf_matrix <- table(Actual = test_data$Success_Faliure, Predicted = predicted_classes)
sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
specificity <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
print(paste("Sensitivity (True Positive Rate):", sensitivity))
print(paste("Specificity (True Negative Rate):", specificity))
total_correct <- sum(diag(conf_matrix))
total_predictions <- sum(conf_matrix)
accuracy <- total_correct / total_predictions
print(paste("Accuracy:", accuracy))
print("Confusion Matrix:")
print(conf_matrix)

#Question 3
#A
# Create logistic regression model
log_model <- glm(success_failure ~ has_item_song, data = bollywood, family = binomial)
summary(log_model)

# Descriptive analytics
# Compute success probabilities for movies with and without item songs
success_prob_with <- mean(predict(log_model, type = "response")[bollywood$has_item_song == 1])
success_prob_without <- mean(predict(log_model, type = "response")[bollywood$has_item_song == 0])

# Print the success probabilities
cat("Success probability for movies with item songs:", success_prob_with, "\n")
cat("Success probability for movies without item songs:", success_prob_without, "\n")

# Visualization
# Create a bar plot to visualize the difference in success probabilities
success_prob_data <- data.frame(Item_Song = c("With Item Song", "Without Item Song"),
                                Success_Probability = c(success_prob_with, success_prob_without))

ggplot(success_prob_data, aes(x = Item_Song, y = Success_Probability, fill = Item_Song)) +
  geom_bar(stat = "identity") +
  labs(x = "Item Song", y = "Success Probability", title = "Success Probability of Movies with and without Item Songs") +
  theme_minimal()

#B
model_budget <- glm(success_failure ~ budget, data = bollywood, family = binomial)
summary(model_budget)
model_item_song <- glm(success_failure ~ has_item_song, data = bollywood, family = binomial)
summary(model_item_song)

# Predictions using the model with Budget as the independent variable
pred_budget <- ifelse(predict(model_budget, type = "response") > 0.5, 1, 0)

# Predictions using the model with Item Song as the independent variable
pred_item_song <- ifelse(predict(model_item_song, type = "response") > 0.5, 1, 0)

# Create confusion matrices
conf_matrix_budget <- table(pred_budget, bollywood$success_failure)
conf_matrix_item_song <- table(pred_item_song, bollywood$success_failure)

# Compare confusion matrices
print("Confusion Matrix for Model with Budget:")
print(conf_matrix_budget)
print("Confusion Matrix for Model with Item Song:")
print(conf_matrix_item_song)

# Calculate accuracy for each model
accuracy_budget <- sum(diag(conf_matrix_budget)) / sum(conf_matrix_budget)
accuracy_item_song <- sum(diag(conf_matrix_item_song)) / sum(conf_matrix_item_song)

# Print accuracy
print(paste("Accuracy for Model with Budget:", round(accuracy_budget, 3)))
print(paste("Accuracy for Model with Item Song:", round(accuracy_item_song, 3)))

#Question 4
# Logistic regression model with all relevant variables
logit_model_req <- glm(success_failure ~ youtube_likes + box_office_collection + has_item_song + lead_actor_category + music_director_category + director_category + production_house_category + profit + earning_ratio + budget, data = train_data, family = 'binomial')

# Calculating VIF to check for multicollinearity
vif_values <- car::vif(logit_model_req)

# Print VIF values
print(vif_values)

# Plot VIF values
plot(vif_values, type = "l", main = "VIF Values for Independent Variables", xlab = "Independent Variables", ylab = "VIF")
plot(vif_values, type = "b", main = "VIF Values for Independent Variables", xlab = "VIF", ylab = "Independent Variables")
plot(vif_values, type = "b", main = "VIF Values for Independent Variables", xlab = "Independent Variables", ylab = "VIF")

# Summary of logistic regression model with all relevant variables
summary(logit_model_req)

# Prediction using the model
predict_req <- predict(logit_model_req, test_data, type = "response")
predict_r <- factor(ifelse(predict_req > 0.5, "1", "0"), levels = c("0", "1"))
test_data$success_failure <- factor(test_data$success_failure, levels = c("0", "1"))
conf_matrix_all <- confusionMatrix(predict_r, test_data$success_failure, positive = "1", mode = "prec_recall")
print(conf_matrix_all)

# Logistic regression model with all important variables
logit_model_all <- glm(success_failure ~ youtube_likes + box_office_collection + has_item_song + lead_actor_category + music_director_category + director_category + production_house_category, data = train_data, family = 'binomial')

# Calculating VIF to check for multicollinearity
vif_values <- car::vif(logit_model_all)

# Print VIF values
print(vif_values)

# Plot VIF values
plot(vif_values, type = "l", main = "VIF Values for Independent Variables", xlab = "Independent Variables", ylab = "VIF")
plot(vif_values, type = "b", main = "VIF Values for Independent Variables", xlab = "VIF", ylab = "Independent Variables")
plot(vif_values, type = "b", main = "VIF Values for Independent Variables", xlab = "Independent Variables", ylab = "VIF")

# Summary of logistic regression model with all important variables
summary(logit_model_all)

# Prediction using the model
predict_all <- predict(logit_model_all, test_data, type = "response")
predict_a <- factor(ifelse(predict_all > 0.5, "1", "0"), levels = c("0", "1"))
test_data$success_failure <- factor(test_data$success_failure, levels = c("0", "1"))
conf_matrix_all <- confusionMatrix(predict_a, test_data$success_failure, positive = "1", mode = "prec_recall")
print(conf_matrix_all)

