####################################################################################
# Name: GSMARENA url scraping 
# Description: Get the urls of phones in gsmarena.com for mobile atlas table creation
# Version:
#   2016/10/03 RS: Initial version
#   
####################################################################################


# install.packages("xml2")
# install.packages("stringr")
library(rvest)
library(xml2)
library(ggplot2)
library(data.table)
library(plyr)
library(bitops)
library(stringr)

# Clear objects
rm(list=ls())

# Set default directory
default_directory <- "D:/Tapad_UC1/Mobile_Atlas/Device_Clustering/RESULTS"
#default_directory <- "R:/ShareData/Rata/Mobile_Atlas"

setwd(default_directory)


# Load the page
gsmarena.mainpage <- html(x = "http://www.gsmarena.com/makers.php3")

# Extract all the urls on the page
urls.brand <- gsmarena.mainpage %>% # feed `main.page` to the next step
  html_nodes(".st-text a") %>% # get the CSS nodes
  html_attr("href") # extract the URLs

# Keep only the urls related to phone and create a vector of phone brands
urls.brand <- urls.brand[grep("phone",urls.brand)]
urls.brand.split <- strsplit(urls.brand,"-")

brand.list <- data.frame(matrix(NA, nrow = length(urls.brand), ncol = 1))
for (k in 1:length(urls.brand)){
  brand.list[k,1] <- urls.brand.split[[k]][1]
}

brand.list <- brand.list[,1]

# Write the urls of the "brand" page

urls.brand <- apply(data.frame(urls.brand),1, function(x) paste(c("http://www.gsmarena.com/",x),collapse=''))

# Scrape the urls of the phone from each brand & Create a csv file of the urls per brand 

for (k in (1:length(urls.brand))){
  print(brand.list[k])
  main.page <- urls.brand[k]
  html <- paste(readLines(main.page), collapse="\n")
  allhref <- str_match_all(html, "<a href=\"(.*?)\"")
  
  # Scrape the urls of the brand page, e.g. "acer-phones-f-59-0-p2.php"
  
  urls.brand.page <-  data.frame(allhref[[1]][ ,2][grepl("-p[1-9]",allhref[[1]][ ,2])]) # Extract any url having "-p1" ... "-p9"
  
  urls.brand.page <- apply(urls.brand.page,1, function(x) paste(c("http://www.gsmarena.com/",x),collapse=''))
  
  # Scrape the urls of the phone on each brand page
  
  for (j in 1:length(urls.brand.page)){
    
    main.brand.page <- urls.brand.page[j]
    html <- paste(readLines(main.brand.page), collapse="\n")
    # extract all the urls on the page
    allmodel <- str_match_all(html, "<a href=\"(.*?)\"")
    
    # choose only the phone urls corresponding to the brand page & create a dataframe of the urls
    
    matched <- allmodel[[1]][ ,2][grepl(paste(c(brand.list[k],"_"),collapse=''),allmodel[[1]][ ,2])&!grepl("review",allmodel[[1]][ ,2])]
    urls <- data.frame(matrix(NA, nrow = length(matched), ncol = 1))
    matched <- data.frame(matched)
    urls <- apply(matched,1, function(x) paste(c("http://www.gsmarena.com/",x),collapse=''))
    
    if (j == 1){
      urls_brand <- data.frame(urls)  
    } else {
      urls_brand <- rbind(urls_brand,data.frame(urls))
    }
    
  }
  
  
  nameurls.file <- paste("URLS", toupper(brand.list[k]), sep="_")
  assign(nameurls.file,urls_brand)
  write.csv(urls_brand, file=paste(nameurls.file,".csv",sep=""), row.names = FALSE)

  # Create a dataframe of the urls for all the brands
  if (k==1){
    URLS_ALLMODEL <- data.frame(urls_brand)  
  } else {
    URLS_ALLMODEL <- rbind(URLS_ALLMODEL,urls_brand)
  }
  
  
}

write.csv(URLS_ALLMODEL, file="URLS_ALLMODEL.csv", row.names = FALSE)


