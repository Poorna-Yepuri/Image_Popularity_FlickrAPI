# Clean Data 
rm(list=ls())

# Flickr Library
#install.packages("FlickrAPI")
#library(FlickrAPI)
library(devtools)
#devtools::install_github("nfox29/photosearcher", force = TRUE)
#devtools::install_github("Poorna-Yepuri/FlickrAPI", force = TRUE)
library(data.table)
library(dplyr)
library(tidyr)

#e45d403b140e48b487cffa7601fabdfc
#4dbfccfb4853b4cfa93325c1ecd1cf63

# Setting Flickr API Key
FlickrAPI::setFlickrAPIKey(api_key = "e45d403b140e48b487cffa7601fabdfc", install = TRUE, overwrite = TRUE)
readRenviron("~/.Renviron") # Read environment variables.

# Source Images from USA region 
geo_bbox <- c(-171.791110603, 18.91619, -66.96466, 71.3577635769)

# Getting image files for interesting and not interesting score of Flickr algorithm

pages <- c(1:140) 
sort_types <- c("relevance", "interestingness-asc", "interestingness-desc")
# 140 pages X 250 = 35,000 image records for each sort type

for (sort_method in sort_types){
  count <- 1 # For first time creation of dataframe
  
for (page_num in pages){
image_scrape <- FlickrAPI::getPhotoSearch(sort = sort_method,
                             bbox = geo_bbox,
                             img_size = "n",
                             extras = c("description", "license", "date_upload", "date_taken", "owner_name",
                               "icon_server", "original_format", "last_update", "geo", "tags", "machine_tags",
                               "o_dims", "views", "media", "path_alias", "url_n", "count_views","count_comments","count_faves"),
                             per_page = 500,
                             page = page_num
)

image_scrape$description <- image_scrape$description[[1]]

ifelse (count == 1, df_scrape <- rbind(image_scrape), df_scrape <- rbind(df_scrape, image_scrape))
count <- 0


}
  myfile <- file.path(getwd(), paste0(sort_method, "_image_info", ".csv"))
  fwrite(df_scrape, file = myfile)
  
  # for emptying df_scrape dataframe
  df_empty <- data.frame(matrix(ncol = ncol(df_scrape), nrow = 0))
  columns_list <- colnames(df_scrape)
  colnames(df_empty) <- columns_list
  df_scrape <- df_empty
}

##### Code can be started here with scraped information #####

relevance_df <- read.csv("./relevance_image_info.csv", header = TRUE)
interestingness_asc_df <- read.csv("./interestingness-asc_image_info.csv", header = TRUE)
interestingness_desc_df <- read.csv("./interestingness-desc_image_info.csv", header = TRUE)


fols <- list()
curr_dir <- getwd()
for (fol in c("relevance", "int_asc", "int_desc")){
  fol_path <- file.path(curr_dir,fol)
  fols <- append(fols, fol_path)
  dir.create(paste0(fol_path))
}

relevance_df$image_name <- paste0(relevance_df$id, "_", relevance_df$secret)
interestingness_asc_df$image_name <- 
  paste0(interestingness_asc_df$id, "_", interestingness_asc_df$secret)
interestingness_desc_df$image_name <- 
  paste0(interestingness_desc_df$id, "_", interestingness_desc_df$secret)

#relevance_df <- relevance_df %>% drop_na("o_width", "o_height")
#interestingness_asc_df <- interestingness_asc_df %>% drop_na("o_width", "o_height")
#interestingness_desc_df <- interestingness_desc_df %>% drop_na("o_width", "o_height")

relevance_df <- relevance_df[!duplicated(relevance_df$image_name),]
interestingness_asc_df <- interestingness_asc_df[!duplicated(interestingness_asc_df$image_name),]
interestingness_desc_df <- interestingness_desc_df[!duplicated(interestingness_desc_df$image_name),]


## checking for previously downloaded Relevance files
rel_images <- list.files("./relevance/")

rel_images_id <- list()
for (imgs in rel_images){
  img_split <- strsplit(imgs, split = "_")
  rel_images_id <- append(rel_images_id, img_split[[1]][1])
}

#, interestingness_asc_df, interestingness_desc_df
for (fol in list(relevance_df)){
  photo_ids <- fol$id
  for (id in photo_ids){
    if (id %in% rel_images_id){
      paste("Already Present")
      next
    }
    tryCatch( photosearcher::download_images(
      id,
      save_dir = paste0(fols[[1]]),
      max_image_height = 500,
      max_image_width = 500,
      overwrite_file = TRUE,
      quiet = FALSE
    ), error = function(err) {paste("Skipped File")})
  }
}


asc_images <- list.files("./int_asc/")

asc_images_id <- list()
for (imgs in asc_images){
  img_split <- strsplit(imgs, split = "_")
  asc_images_id <- append(asc_images_id, img_split[[1]][1])
}


for (fol in list(interestingness_asc_df)){
  photo_ids <- fol$id
  for (id in photo_ids){
    if (id %in% asc_images_id){
      paste("Already Present")
      next
    }
    tryCatch( photosearcher::download_images(
      id,
      save_dir = paste0(fols[[2]]),
      max_image_height = 500,
      max_image_width = 500,
      overwrite_file = TRUE,
      quiet = FALSE
    ), error = function(err) {paste("Skipped File")})
  }
}



desc_images <- list.files("./int_desc/")

desc_images_id <- list()
for (imgs in desc_images){
  img_split <- strsplit(imgs, split = "_")
  desc_images_id <- append(desc_images_id, img_split[[1]][1])
}


for (fol in list(interestingness_desc_df)){
  photo_ids <- fol$id
  for (id in photo_ids){
    if (id %in% desc_images_id){
      paste("Already Present")
      next
    }
    tryCatch( photosearcher::download_images(
      id,
      save_dir = paste0(fols[[3]]),
      max_image_height = 500,
      max_image_width = 500,
      overwrite_file = TRUE,
      quiet = FALSE
    ), error = function(err) {paste("Skipped File")})
  }
}



