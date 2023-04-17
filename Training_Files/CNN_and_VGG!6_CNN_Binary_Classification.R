#### Classification of popularity categories based on the Trained CNN Regression Model ####

test_results <- read.csv("./test_results_1.csv", header = TRUE)
class_df <- transform(test_results, Pop_class_actual = ifelse(test_results$test_ds.labels > 200, "Popular", "Unpopular"),
                      Pop_class_pred = ifelse(test_results$pred > 200, "Popular", "Unpopular"))

matrix_conf <- table(class_df$Pop_class_actual, class_df$Pop_class_pred)

class_acc <- (matrix_conf[[1,1]]+matrix_conf[[2,2]]) / nrow(test_results)

fwrite(class_df, file = file.path(getwd(), paste0("class_pred.csv")))
#### Classification of Popularity categories based model built on VGG-16 pre-trained net ####

vgg_test_results <- read.csv("./vgg_test_results_1.csv", header = TRUE)
colnames(vgg_test_results)[c(1,2,3)] <- c("vgg_image_name", "vgg_test_views_lables", "vgg_pred")
vgg_class_df <- transform(vgg_test_results, Pop_class_actual = ifelse(vgg_test_results$vgg_test_views_lables > 200, "Popular", "Unpopular"),
                      Pop_class_pred = ifelse(vgg_test_results$vgg_pred > 200, "Popular", "Unpopular"))

vgg_matrix_conf <- table(vgg_class_df$Pop_class_actual, vgg_class_df$Pop_class_pred)

vgg_class_acc <- (vgg_matrix_conf[[1,1]]+vgg_matrix_conf[[2,2]]) / nrow(vgg_test_results)




#### CNN Binary Classification####

final_data_df <- read.csv("./Final_clean_images_info.csv", header = TRUE)
train_df <- read.csv("./train_image_info.csv", header = TRUE)
test_df <- read.csv("./test_image_info.csv", header = TRUE)

binary_train_df <- transform(train_df, Pop_class = ifelse(train_df$count_views > 200, "Popular", "Unpopular"))
binary_test_df <- transform(test_df, Pop_class = ifelse(test_df$count_views > 200, "Popular", "Unpopular"))

# Building CNN model
batch_size <- 50
datagen <- image_data_generator(rescale = 1/255, validation_split = 0.15)

bin_train_ds <- flow_images_from_dataframe(dataframe = binary_train_df, directory = "./train_dir/", 
                                       x_col = "image_name", y_col = "Pop_class",
                                       generator = datagen, target_size = c(150, 150),
                                       color_mode = "rgb", batch_size = batch_size,
                                       class_mode = "binary", subset = "training",
                                       seed = 8, shuffle = FALSE)

bin_validate_ds <- flow_images_from_dataframe(dataframe = binary_train_df, directory = "./train_dir/", 
                                          x_col = "image_name", y_col = "Pop_class",
                                          generator = datagen, target_size = c(150, 150),
                                          color_mode = "rgb", batch_size = 15,
                                          class_mode = "binary", subset = "validation",
                                          seed = 8, shuffle = FALSE)

test_datagen <- image_data_generator(rescale = 1/255)
bin_test_ds <- flow_images_from_dataframe(dataframe = binary_test_df, directory = "./test_dir/", 
                                      x_col = "image_name", y_col = "Pop_class",
                                      generator = test_datagen, target_size = c(150, 150),
                                      color_mode = "rgb", batch_size = 10,
                                      class_mode = "binary", shuffle = FALSE)

binary_class_model <- keras_model_sequential() %>%
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
  layer_dense(units = 1, activation = "sigmoid")

binary_class_model %>% compile(optimizer = "Adam",
                       loss = "binary_crossentropy",
                       metrics = c("Accuracy"))


summary(binary_class_model)

binary_class_history <- binary_class_model %>%
  fit(bin_train_ds, 
      epochs=20, 
      validation_data = bin_validate_ds
  )

plot(binary_class_history)

bin_test_results <- binary_class_model %>%
  evaluate(bin_test_ds, steps = 20)

binary_class_model %>% save_model_hdf5("bin_class_model_07_12_1.h5")
write_rds(binary_class_history, "bin_class_model_07_12_1.rds")
write_rds(bin_test_ds, "bin_class_test_img_gen.rds")


binary_class_model <- load_model_hdf5("bin_class_model_07_12_1.h5")
binary_class_history <- read_rds("bin_class_model_07_12_1.rds")
bin_test_ds <- read_rds("bin_class_test_img_gen.rds")

bin_test_results <- binary_class_model %>%
  evaluate(bin_test_ds, steps = 20)
bin_test_results

bin_test_metrics <- bin_test_results
bin_test_metrics <- data.frame(list(bin_test_metrics))
bin_test_metrics$metric <- row.names(bin_test_metrics)
colnames(bin_test_metrics)[c(1)] <- c("Value")
bin_test_metrics <- bin_test_metrics %>% select(metric, everything())

fwrite(bin_test_metrics, file = file.path(getwd(), paste0("CNN_Class_test_metrics.csv")))


#### CNN VGG-16 Binary Classification
vgg_train_dir <- paste0(curr_dir,"/vgg/vgg_train")
vgg_test_dir <- paste0(curr_dir,"/vgg/vgg_test")
vgg_valid_dir <- paste0(curr_dir,"/vgg/vgg_valid")

vgg_train_df <- read.csv("./vgg_train_data.csv", header = TRUE)
vgg_test_df <- read.csv("./vgg_test_data.csv", header = TRUE)
vgg_valid_df <- read.csv("./vgg_train_data.csv", header = TRUE)

bin_vgg_train_df <- transform(vgg_train_df, Pop_class = ifelse(vgg_train_df$count_views > 200, "Popular", "Unpopular"))
bin_vgg_valid_df <- transform(vgg_valid_df, Pop_class = ifelse(vgg_valid_df$count_views > 200, "Popular", "Unpopular"))
bin_vgg_test_df <- transform(vgg_test_df, Pop_class = ifelse(vgg_test_df$count_views > 200, "Popular", "Unpopular"))


vgg_datagen <- image_data_generator(rescale = 1/255)
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)
conv_base

extract_features <- function(directory, dataframe, sample_count, batch_size) {
  features <- array(0, dim = c(sample_count, 4, 4, 512))
  labels <- array(0, dim = c(sample_count))
  generator <- flow_images_from_dataframe(dataframe = dataframe,
                                          directory = directory, x_col = "image_name", y_col = "Pop_class", generator = vgg_datagen,
                                          target_size = c(150, 150), batch_size = batch_size, class_mode = "binary", seed = 8, shuffle = FALSE
  )
  i <- 0
  while(TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]; labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    i <- i + 1
    if (i * batch_size >= sample_count) break
  }
  return(list(features = features, labels = labels))
}

bin_vgg_train <- extract_features(vgg_train_dir, bin_vgg_train_df, 2730, 30) # will take a while since we are running
bin_vgg_validation <- extract_features(vgg_valid_dir, bin_vgg_valid_df,480, 15) # our images through conv_base
bin_vgg_test <- extract_features(vgg_test_dir, bin_vgg_test_df,800, 20) # still faster than training such a model

#### Last Layer of conv_base collected

reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}
bin_vgg_train$features <- reshape_features(bin_vgg_train$features)
bin_vgg_validation$features <- reshape_features(bin_vgg_validation$features)
bin_vgg_test$features <- reshape_features(bin_vgg_test$features)


### Model Building

bin_vgg_model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu",
              input_shape = 4 * 4 * 512) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

bin_vgg_model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("Accuracy")
)

summary(bin_vgg_model)

history_bin_vgg <- bin_vgg_model %>% fit(
  bin_vgg_train$features, bin_vgg_train$labels,
  epochs = 30,
  validation_data = list(bin_vgg_validation$features, bin_vgg_validation$labels)
)

bin_vgg_model %>% save_model_hdf5("bin_vgg_model_07_12_1.h5")
write_rds(history_bin_vgg, "bin_vgg_model_07_12_1.rds")
write_rds(bin_vgg_test, "bin_VGG-class_test_img_gen.rds")

bin_vgg_model <- load_model_hdf5("bin_vgg_model_07_12_1.h5")
history_bin_vgg <- read_rds("bin_vgg_model_07_12_1.rds")
bin_vgg_test <- read_rds("bin_VGG-class_test_img_gen.rds")

bin_vgg_test_metrics <- bin_vgg_model %>%
  evaluate(bin_vgg_test$features, bin_vgg_test$labels, steps = 20)

bin_vgg_test_metrics





