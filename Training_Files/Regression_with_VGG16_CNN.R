# Utilizing pretrained VGG16 model 
#######################################
dir.create(paste0(curr_dir, "/vgg"))
vgg_train_dir <- paste0(curr_dir,"/vgg/vgg_train")
dir.create(vgg_train_dir)
vgg_test_dir <- paste0(curr_dir,"/vgg/vgg_test")
dir.create(vgg_test_dir)
vgg_valid_dir <- paste0(curr_dir,"/vgg/vgg_valid")
dir.create(vgg_valid_dir)


vgg_train_images <- list()
vgg_valid_images <- list()
vgg_test_images <- list()
vgg_data <- sample.split(train_df$image_name %>% unlist(), SplitRatio = 0.85)
vgg_train_images <- list.append(vgg_train_images, subset(train_df$image_name %>% unlist,vgg_data==TRUE))%>% unlist()
vgg_valid_images <- list.append(vgg_valid_images, subset(train_df$image_name %>% unlist,vgg_data==FALSE)) %>% unlist()

vgg_train_files <- list(paste0(("train_dir/"), vgg_train_images))
file.copy((vgg_train_files %>% unlist()), vgg_train_dir)

vgg_valid_files <- list(paste0(("train_dir/"), vgg_valid_images))
file.copy((vgg_valid_files %>% unlist()), vgg_valid_dir)

vgg_train_df <- combined_df[combined_df$image_name %in% vgg_train_images,]
vgg_valid_df <- combined_df[combined_df$image_name %in% vgg_valid_images,]

vgg_test_files <-list(paste0("test_dir/", test_df$image_name))
file.copy((vgg_test_files %>% unlist()), vgg_test_dir)

vgg_test_df <- test_df

fwrite(vgg_train_df, file = file.path(getwd(), paste0("vgg_train_data.csv")))
fwrite(vgg_valid_df, file = file.path(getwd(), paste0("vgg_valid_data.csv")))
fwrite(vgg_test_df, file = file.path(getwd(), paste0("vgg_test_data.csv")))

##########################################################################


library(keras)

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
    directory = directory, x_col = "image_name", y_col = "count_views", generator = vgg_datagen,
    target_size = c(150, 150), batch_size = batch_size, class_mode = "raw", seed = 8, shuffle = FALSE
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

train <- extract_features(vgg_train_dir, vgg_train_df, 2730, 30) # will take a while since we are running
validation <- extract_features(vgg_valid_dir, vgg_valid_df,480, 15) # our images through conv_base
test <- extract_features(vgg_test_dir, vgg_test_df,800, 20) # still faster than training such a model

#### Last Layer of conv_base collected

reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}
train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)


### Model Building

vgg_model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu",
              input_shape = 4 * 4 * 512) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

vgg_model %>% compile(
  optimizer = "adam",
  loss = "mse",
  metrics = c("MeanSquaredError", "RootMeanSquaredError", "MeanAbsoluteError")
)

summary(vgg_model)

history_vgg <- vgg_model %>% fit(
  train$features, train$labels,
  epochs = 30,
  validation_data = list(validation$features, validation$labels)
)

vgg_model %>% save_model_hdf5("vgg_model_07_12_1.h5")
write_rds(history_vgg, "vgg_model_07_12_1.rds")
write_rds(test, "VGG-CNN_Reg_Test_img_gen.rds")

vgg_model <- load_model_hdf5("vgg_model_07_12_1.h5")
history_vgg <- read_rds("vgg_model_07_12_1.rds")
test <- read_rds("VGG-CNN_Reg_Test_img_gen.rds")

vgg_test_metrics <- vgg_model %>%
  evaluate(test$features, test$labels, steps = 20)

vgg_test_metrics

vgg_pred <- vgg_model$predict(test$features)
vgg_test_results <- data.frame(vgg_test_df$image_name[1:800], vgg_test_df$count_views[1:800], vgg_pred)

fwrite(vgg_test_results, file = file.path(getwd(), paste0("vgg_test_results_1.csv")))
  
hist(vgg_test_results$vgg_pred,
     xlab = "views",
     main = "Distribution of Views",
     breaks = nrow(vgg_test_results)/20)
