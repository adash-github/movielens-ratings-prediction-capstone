## ----download, message=FALSE, warning=FALSE, echo=FALSE---------------------------------------------
# Packages we may need to run the code
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(tidyr)) install.packages("tidyr")
if(!require(stringr)) install.packages("stringr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(forcats)) install.packages("forcats")
if(!require(dplyr)) install.packages("dplyr")
if(!require(magrittr)) install.packages("magrittr")

library(tidyverse)
library(caret)
library(data.table)
library(kableExtra)
library(tidyr)
library(stringr)
library(ggplot2)
library(forcats)
library(dplyr)
library(magrittr)


# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Here we download the data and pre-process it 

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")


## ----message=FALSE, warning=FALSE, echo=TRUE--------------------------------------------------------
# Validation set will be 10% of MovieLens data
set.seed(1) # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)



## ----message=FALSE, warning=FALSE, echo=TRUE--------------------------------------------------------
##### Split Edx Again #####

# This is to avoid over-training, we want to split our set again. 

set.seed(1)
edx_test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-edx_test_index,]
edx_temp <- edx[edx_test_index,]

# Make sure userId and movieId in edx_validation set are also in edx_train set
edx_validation <- edx_temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Add rows removed from validation set back into edx set
edx_removed <- anti_join(edx_temp, edx_validation)
edx_train <-  rbind(edx_train, edx_removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



## ----warning=FALSE, echo=FALSE----------------------------------------------------------------------
print("edx_train:")
head(edx_train)
summary(edx_train)
print("edx_validation:")
head(edx_validation)
summary(edx_validation)


## ----warning=FALSE----------------------------------------------------------------------------------
##### Base Mean Model #####

# First create data frame to store our RMSE Results

rmse_results <- data_frame()

# Base Model
# Just using the mean
mu <- mean(edx_train$rating)
mu

base_rmse <- RMSE(edx_validation$rating, mu)
rmse_results <- data_frame(Model = "Base Mean Model", RMSE = base_rmse)

rmse_results %>% knitr::kable() # To look at our results



## ----warning=FALSE, echo=FALSE----------------------------------------------------------------------
### Movie Effect Model ###

movie_avg <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

qplot(b_i, data = movie_avg, bins = 25, color = I("black")) # generate plot of movie bias


## ----warning=FALSE, echo=TRUE-----------------------------------------------------------------------
pred_ratings <- mu + edx_validation %>%
  left_join(movie_avg, by = 'movieId') %>%
  pull(b_i)

movie_rmse <- RMSE(edx_validation$rating, pred_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="Movie Effect Model",  
                                     RMSE = movie_rmse))
rmse_results %>% knitr::kable()


## ----echo=FALSE-------------------------------------------------------------------------------------
# Plot to see how our users breakdown

edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 35, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users and Number of Reviews/Ratings") +
  xlab("Number of Reviews/Ratings") +
  ylab("Frequency by User")


## ----echo=FALSE-------------------------------------------------------------------------------------
##### User + Movie Model #####

user_avg <- edx_train %>%
  left_join(movie_avg, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

qplot(b_u, data = user_avg, bins = 25, color = I("black")) # generate plot of user bias



## ----echo=TRUE--------------------------------------------------------------------------------------
pred_ratings_user <- edx_validation %>%
  
  left_join(movie_avg, by = 'movieId') %>%
  left_join(user_avg, by = 'userId') %>%
  mutate(prediction = mu + b_i + b_u) %>%
  pull(prediction)

model_user_movie <- RMSE(edx_validation$rating, pred_ratings_user)
rmse_results <- bind_rows(rmse_results, 
                          data_frame(Model = "Movie and User Effect Model",
                                     RMSE = model_user_movie))

rmse_results %>% knitr::kable()


## ----echo=TRUE--------------------------------------------------------------------------------------
### WARNING: This step takes a long time
# This is to seperate the genres out into individual tags to then graph the mean rating of each genre

edx_genre <- edx %>%
   mutate(genre = fct_explicit_na(genres,
                                       na_level = "(no genres listed)")
          ) %>%
   separate_rows(genre,
                 sep = "\\|")


## ---------------------------------------------------------------------------------------------------
# Here we access the dataset we created with the individual genres and calculatate the means for each genre

edx_genre %>%
   group_by(genre) %>%
   summarise(mean = mean(rating)) %>%
   ggplot(aes(genre, mean)) +
   theme_classic()  +
   geom_col() +
   theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
   labs(title = "Means of each Genre",
        x = "Genres",
        y = "Mean")




## ----echo=FALSE-------------------------------------------------------------------------------------
# Create bar plot to show genre breakdown by percentage
perc_gen <- edx_genre %>%
  group_by(genre) %>%
  summarize(n=n()) %>%
  ungroup() %>%
  mutate(sumG = sum(n), percentage = n/sumG) %>%
  arrange(-percentage)

perc_gen %>%
  ggplot(aes(reorder(genre, percentage), percentage, fill= percentage)) +
  geom_bar(stat = "identity") + coord_flip() +
  xlab("Genre") +
  ylab("Percentage") +
  ggtitle("Genre Distribution by Percent Rated")


## ----echo=TRUE--------------------------------------------------------------------------------------
##### Movie, User, and Genre Model #####
genre_avg <- edx_train %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

pred_ratings_genre <- edx_validation %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(genre_avg, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

model_user_movie_genre <- RMSE(edx_validation$rating, pred_ratings_genre)
rmse_results <- bind_rows(rmse_results, 
                          data_frame(Model = "Movie, User. and Genre Effect Model",
                                     RMSE = model_user_movie_genre))

rmse_results %>% knitr::kable()



## ----echo=TRUE--------------------------------------------------------------------------------------
# Generate a Lambda plot to find our lowest RMSE

lamdas <- seq(0, 10, 0.25)

# Create a function to generate lambdas and their respective RMSE value
# Will use the "equation" again to double check the RMSE with our found lamda later

rmses <- sapply(lamdas, function(l){
  mu <- mean(edx_train$rating)
  
  b_i <- edx_train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l))
  
  b_u <- edx_train %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)  / (n() + l))  
  
  b_g <- edx_train %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u) / (n() + l), n_g = n())
  
  predicted_ratings <- edx_validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    .$pred
  
  return(RMSE(edx_validation$rating, predicted_ratings))
  
})

qplot(lamdas, rmses) # Plot to see the shape of graph and estimate our lowest RMSE

min_lamda <- lamdas[which.min(rmses)] # assigns lowest RMSE so we can use it later
min_lamda

##### Find RMSE using Lamda Value #####

# Using the lambda value generated with the estimated lowest RMSE we can actually plug it back in

mu <- mean(edx_train$rating)

b_i <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + min_lamda))

b_u <- edx_train %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)  / (n() + min_lamda))

b_g <- edx_train %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u) / (n() + min_lamda), n_g = n())

reg_predicted_ratings <- edx_validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred

reg_RMSE <- RMSE(edx_validation$rating, reg_predicted_ratings)

rmse_results <- bind_rows(rmse_results, 
                          data_frame(Model = "Reg: Movie, User. and Genre Effect Model",
                                     RMSE = reg_RMSE))
rmse_results %>% knitr::kable()


## ----echo=TRUE--------------------------------------------------------------------------------------
##### Final Validation Test #####

# Using the lamda value we found we remove the function and just see what our RMSE value is
# This time we will be using our final validation set to see what our lowest RMSE is

mu <- mean(edx$rating)

b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + min_lamda))

b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)  / (n() + min_lamda))

b_g <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u) / (n() + min_lamda), n_g = n())

# Utilizing final Validation set to see what our RMSE is

final_reg_predicted_ratings <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  .$pred

final_reg_RMSE <- RMSE(validation$rating, final_reg_predicted_ratings)

rmse_results <- bind_rows(rmse_results, 
                          data_frame(Model = "Final Validation",
                                     RMSE = final_reg_RMSE))
rmse_results %>% knitr::kable()

