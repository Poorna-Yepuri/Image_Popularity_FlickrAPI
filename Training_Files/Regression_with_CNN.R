rm(list=ls())

#install.packages(c("keras", "reticulate"))
#install.packages("tensorflow")
# Libraries
library(tidyverse)
library(keras)
library(tensorflow)
library(rlist)
library(caTools)
library(data.table)
library(ggplot2)

curr_dir <- getwd()
# Can directly start from line 170

###################################################
## Relevance_Images: Matching Flickr Data with Downloaded Images Data
imgs_dir <- paste0(curr_dir, "/relevance/")
imgs_dataset <- c(list.files(imgs_dir))
imgs_dataset <- as.vector(imgs_dataset)

relevance_df <- read.csv("./relevance_image_info.csv", header = TRUE)

relevance_df$image_name <- paste0(relevance_df$id, "_", relevance_df$secret, ".jpg")

# only keeping the data for images that are downloaded.
downs_imgs_df <-relevance_df[relevance_df$image_name %in% imgs_dataset,]
downs_imgs_df <- downs_imgs_df[!duplicated(downs_imgs_df$image_name),]


unlisted_files <- list()
for (x in imgs_dataset){
  if (!(x %in% downs_imgs_df$image_name)){ 
    unlisted_files<- append(unlisted_files, x)
  }
  }
unlink(paste0("relevance/",unlisted_files), recursive = FALSE)

relevance_df <- downs_imgs_df[,c("id", "image_name", "count_views", "count_faves", "count_comments")]
combined_df <- relevance_df


###############################################################################
## Interesting_desc_Images: Matching Flickr Data with Downloaded Images Data

imgs_dir <- paste0(curr_dir, "/int_desc/")
imgs_dataset <- c(list.files(imgs_dir))
imgs_dataset <- as.vector(imgs_dataset)

interestingness_desc_df <- read.csv("./interestingness-desc_image_info.csv", header = TRUE)

interestingness_desc_df$image_name <- paste0(interestingness_desc_df$id, "_", interestingness_desc_df$secret, ".jpg")

# only keeping the data for images that are downloaded.
downs_imgs_df <-interestingness_desc_df[interestingness_desc_df$image_name %in% imgs_dataset,]
downs_imgs_df <- downs_imgs_df[!duplicated(downs_imgs_df$image_name),]


unlisted_files <- list()
for (x in imgs_dataset){
  if (!(x %in% downs_imgs_df$image_name)){ 
    unlisted_files<- append(unlisted_files, x)
  }
}
unlink(paste0("int_desc/",unlisted_files), recursive = FALSE)

interestingness_desc_df <- downs_imgs_df[,c("id", "image_name", "count_views", "count_faves", "count_comments")]

combined_df <- rbind(combined_df, interestingness_desc_df)

#######################################################
## Clean Duplicates in the combined Data
master_dir <- paste0(curr_dir,"/master_dir")
dir.create(master_dir)

## Files from Relevance folder
rel_files <- list(paste0("relevance/",list.files("relevance")))
file.copy((rel_files %>% unlist()), master_dir)

## Files from Int_Desc folder
int_desc_files <- list(paste0("int_desc/", list.files("int_desc")))
file.copy((int_desc_files %>% unlist()), master_dir, overwrite = TRUE)

## Files in Master directory
master_files <- list.files("master_dir/")

combined_df <- combined_df[!duplicated(combined_df$image_name),]

unlisted_files <- list()
for (x in master_files){
  if (!(x %in% combined_df$image_name)){ 
    unlisted_files<- append(unlisted_files, x)
  }
}

unlink(paste0("master_dir/",unlisted_files), recursive = FALSE)
fwrite(combined_df, file = file.path(getwd(), paste0("master_image_info.csv")))

######## Start Here #########
# combined_df <- read.csv("./master_image_info.csv", header = TRUE)

##### Data preprocessing ######
str(combined_df)
summary(combined_df$count_views)
options(scipen=999) # remove scientific notation

hist(combined_df$count_views,
     xlab = "views",
     main = "Distribution of Views",
     breaks = nrow(combined_df)/20)

ggplot(combined_df) +
  aes(x = "", y = count_views) +
  geom_boxplot() +
  theme_minimal()

## Most of the data is left skewed ## Need to drop out of bounds values.
# Max value is at 281266

### Removing outliers using interquartile range
MAX <- max(combined_df$count_views)
Q1 <- quantile(combined_df$count_views, 0.25)
Q3 <- quantile(combined_df$count_views, 0.75)
IQR <- IQR(combined_df$count_views)

final_data_df <- subset(combined_df, combined_df$count_views>(Q1 - 1.5*IQR) & 
                        combined_df$count_views < (Q3 + 1.5*IQR))

hist(final_data_df$count_views,
     xlab = "views",
     main = "Distribution of Views",
     breaks = nrow(final_data_df)/10)

ggplot(final_data_df) +
  aes(x = "", y = count_views) +
  geom_boxplot() +
  theme_minimal()

### Images with views that are within appropriate range.
inlier_images <- final_data_df$image_name

#no_outliers$count_views <- log1p(no_outliers$count_views)

#######################################################
##### Data split into Train & Test Dataset ####
train_dir <- paste0(curr_dir,"/train_dir")
dir.create(train_dir)
test_dir <- paste0(curr_dir,"/test_dir")
dir.create(test_dir)

train_images <- list()
test_images <- list()
data <- sample.split(inlier_images %>% unlist(), SplitRatio = 0.8)
train_images <- list.append(train_images, subset(inlier_images %>% unlist,data==TRUE))%>% unlist()
test_images <- list.append(test_images, subset(inlier_images %>% unlist,data==FALSE)) %>% unlist()

train_files <- list(paste0(("master_dir/"), train_images))
file.copy((train_files %>% unlist()), train_dir)

test_files <- list(paste0(("master_dir/"), test_images))
file.copy((test_files %>% unlist()), test_dir)

train_df <- combined_df[combined_df$image_name %in% train_images,]
test_df <- combined_df[combined_df$image_name %in% test_images,]



#######################################################
#######################################################
######## Can Work From here ########
####### Data Saved for Reuse ######

fwrite(final_data_df, file = file.path(getwd(), paste0("Final_clean_images_info.csv")))
fwrite(train_df, file = file.path(getwd(), paste0("train_image_info.csv")))
fwrite(test_df, file = file.path(getwd(), paste0("test_image_info.csv")))

# final_data_df <- read.csv("./Final_clean_images_info.csv", header = TRUE)
# train_df <- read.csv("./train_image_info.csv", header = TRUE)
# test_df <- read.csv("./test_image_info.csv", header = TRUE)





######## CNN Model #########

batch_size = 80
input_shape = c(150, 150, 3)

## Image Generator ##
datagen <- image_data_generator(rescale = 1/255, validation_split = 0.15)

train_ds <- flow_images_from_dataframe(dataframe = train_df, directory = "./train_dir/", 
                                       x_col = "image_name", y_col = "count_views",
                                       generator = datagen, target_size = c(150, 150),
                                       color_mode = "rgb", batch_size = batch_size,
                                       class_mode = "raw", subset = "training",
                                       seed = 8, shuffle = FALSE)

validate_ds <- flow_images_from_dataframe(dataframe = train_df, directory = "./train_dir/", 
                                       x_col = "image_name", y_col = "count_views",
                                       generator = datagen, target_size = c(150, 150),
                                       color_mode = "rgb", batch_size = 15,
                                       class_mode = "raw", subset = "validation",
                                       seed = 8, shuffle = FALSE)

test_datagen <- image_data_generator(rescale = 1/255)
test_ds <- flow_images_from_dataframe(dataframe = test_df, directory = "./test_dir/", 
                                      x_col = "image_name", y_col = "count_views",
                                      generator = test_datagen, target_size = c(150, 150),
                                      color_mode = "rgb", batch_size = 10,
                                      class_mode = "raw", shuffle = FALSE)



base_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  #layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  #layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")
  
base_model %>% compile(optimizer = "Adam",
                       loss = "mse",
                       metrics = c("MeanSquaredError", "RootMeanSquaredError", "MeanAbsoluteError"))


summary(base_model)




history <- base_model %>%
  fit(train_ds, 
      epochs=20, 
      validation_data = validate_ds
  )

plot(history)

base_model %>% save_model_hdf5("new_model_06_12_1.h5")
write_rds(history, "new_model_06_12_1.rds")
write_rds(test_ds, "CNN_Reg_Test_img_gen.rds")


base_model <- load_model_hdf5("new_model_06_12_1.h5")
history <- read_rds("new_model_06_12_1.rds")
test_ds <- read_rds("CNN_Reg_Test_img_gen.rds")

base_test_metrics <- base_model %>%
  evaluate(test_ds, steps = 20)
base_test_metrics
base_test_metrics <- data.frame(list(base_test_metrics))
base_test_metrics$metric <- row.names(base_test_metrics)
colnames(base_test_metrics)[c(1)] <- c("Value")
base_test_metrics <- base_test_metrics %>% select(metric, everything())

fwrite(base_test_metrics, file = file.path(getwd(), paste0("Base_CNN_Reg_test_metrics.csv")))

pred <- base_model$predict(test_ds)
test_results <- data.frame(test_ds$filenames, test_ds$labels, pred)

fwrite(test_results, file = file.path(getwd(), paste0("test_results_1.csv")))

summary(test_results)
hist(test_results$pred,
     xlab = "views",
     main = "Distribution of Views",
     breaks = nrow(test_results)/20)




