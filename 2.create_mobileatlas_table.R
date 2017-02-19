####################################################################################
# Name: Create Mobile Atlas Table
# Description: Scrape the phone information on gsmarena.com to create a mobile atlas table
# Version:
#   2016/10/04 RS: Initial version
#   
####################################################################################

#install.packages("sqldf")
#install.packages("bitops")

library(rvest)
library(ggplot2)
library(data.table)
library(plyr)
library(bitops)
library(sqldf)

# clear objects
rm(list=ls())

# set default directory
default_directory <- "D:/Tapad_UC1/Mobile_Atlas/Device_Clustering/RESULTS" 
#default_directory <- "R:/ShareData/Rata/Mobile_Atlas"

setwd(default_directory)

# Call the URLs table for each brand
temp = list.files( pattern="URLS_")
namebrandall <- gsub("URLS_","",gsub(".csv.*",'',temp))
namebrandall <- namebrandall[!grepl("ALLMODEL",namebrandall)]
namebrandall  <- data.frame(namebrandall )
colnames(namebrandall) <- 'BRAND'

# In case of having info tables for some brands already
existing <- list.files(pattern="_INFO")
namebrandexisting <- gsub("_INFO.*",'',existing)
namebrandexisting <- data.frame(namebrandexisting)
colnames(namebrandexisting) <- 'BRAND'

name.brand.left<- sqldf("select a.BRAND from namebrandall a LEFT OUTER JOIN namebrandexisting b on a.BRAND = b.BRAND where b.BRAND is null")
name.brand.left <- name.brand.left[,1]

  for (numfile in 1:length(name.brand.left)){  
  print(numfile)
  name.brand <- name.brand.left[numfile]
  print(name.brand)
  ddta<-fread(paste("URLS_",name.brand.left[numfile],".csv",sep=''), integer64 = "character",header=TRUE)
  ddta <- data.frame(ddta)
  
  mobileInfo <- data.frame(matrix(NA, nrow = nrow(ddta), ncol = 6))
  colnames(mobileInfo) <- c("BRAND_GSM","MODEL_GSM","RELEASE_TIME_GSM","PRICE_GSM","SCREEN_GSM","DETAIL_NAME_GSM")
  
  for (k in 1:nrow(ddta)){
    if (k %% 10==0){
      print(k)
    }
    
    webpage <- html(as.character(ddta$urls[k]))
    
    
    # Create brand, model and technical name #####
      
    Name <- webpage  %>% html_node(".specs-phone-name-title") %>%
      html_text()
    dummy <- unlist(strsplit(Name," "))
    brandName <- dummy[1]
    modelName <- paste(dummy[-1],collapse=' ')
    
    technicalName <- matrix(NA, nrow = 1, ncol = 1)
    detailName <- matrix(NA, nrow = 1, ncol = 1)
    
    tryCatch({
      technicalName <- webpage  %>% html_node("#specs-list p") %>%
        html_text() 
    },error=function(e){})
    if (is.na(technicalName)==FALSE){
      
      tryCatch({
        detailName <- webpage  %>% html_nodes("#specs-list p") %>%
          html_text() 
        detailName <- detailName[1]
      },error=function(e){})
      
    }
    
    # Create released date and price at release date #####
    
    
    technicalName_check <- matrix(NA, nrow = 1, ncol = 1)
    tryCatch({
      
      technicalName_check <- webpage  %>% html_nodes("#specs-list p") %>%
        html_text() 
      nline.detail <- length(technicalName_check)
      technicalName_check <- technicalName_check[1]
    },error=function(e){})
    
    if (is.na(technicalName_check)==TRUE){
      nline.detail <- 0
    }
    
    nlinedate <- nline.detail+3
    a <- paste("table:nth-child(",nlinedate,") tr+ tr .nfo",sep="")
    
    # check if the status is discontinue or cancel
    status <- webpage  %>% html_node(a) %>%
      html_text()
    
    releaseMonth <- NA
    releaseYear <- NA
    
    if (status=="Cancelled"){
      mobileInfo[k,1:3] <- c(brandName,modelName,status)
      
    } else if (status=="Discontinued"){
      b <- paste("table:nth-child(",nlinedate,") th~ .nfo")
      date <- webpage  %>% html_node(b) %>%
        html_text()
      
      dummy <- strsplit(date," ")
      dummy <- data.frame(dummy)
      
      
      releaseYear <- as.character(dummy[,1][tail(grep("20|199",dummy[,1]),n=1)])
      releaseMonth <- as.character(dummy[,1][tail(grep("20|199",dummy[,1]),n=1)+1])
      
      if (!(length(releaseYear)==0)){
        
        if (is.na(releaseMonth)==TRUE){
          releaseTime <- releaseYear 
        } else 
          releaseTime <- paste(releaseYear,releaseMonth)
      } else releaseTime <- NA
    } else 
      date <- webpage  %>% html_node(a) %>%
      html_text() # Normal
    
    
    # For the phones whose release date is available
    if (grepl("not officially",tolower(date))==TRUE){
      releaseTime <- date
    } else {
      dummy <- strsplit(date," ")
      dummy <- data.frame(dummy)
      if(length(grep("20|1999",dummy[,1]))==0){
        b <- paste("table:nth-child(",nlinedate,") th~ .nfo")
        date <- webpage  %>% html_node(b) %>%
          html_text()
        dummy <- strsplit(date," ")
        dummy <- data.frame(dummy)
        releaseYear <- as.character(dummy[,1][tail(grep("20|199",dummy[,1]),n=1)])
        releaseMonth <- as.character(dummy[,1][tail(grep("20|199",dummy[,1]),n=1)+1])
        
        
      } else {
        dummy <- strsplit(date," ")
        dummy <- data.frame(dummy)
        releaseYear <- as.character(dummy[,1][tail(grep("20|199",dummy[,1]),n=1)])
        releaseMonth <- as.character(dummy[,1][tail(grep("20|199",dummy[,1]),n=1)+1])
        
      }
      
      
      if (!(length(releaseYear)==0)){
        
        if (is.na(releaseMonth)==TRUE){
          releaseTime <- releaseYear 
        } else 
          releaseTime <- paste(releaseYear,releaseMonth)
      } else releaseTime <- NA
      
    }
    
    
    
    price <- NA
    tryCatch({
      price <- webpage  %>% html_node(".price") %>%
        html_text()
      price <- as.numeric(gsub("[^0-9]", "", price )) # in euro
      
    },error=function(e){})
    
    screensize <- NA
    # Create screensize 
    tryCatch({
      screensizeCha <- webpage  %>% html_node(".help-display .accent") %>%
        html_text() 
      screensize <- gsub("\"", "", screensizeCha )
      if (length(grep("[0-9]",screensize))==0){
        screensize <- NA   
      } else {
        screensize <- as.numeric(screensize)
      }
    },error=function(e){})
    
    
    mobileInfo[k,1:6] <- c(brandName,modelName,releaseTime,price,screensize,detailName)
    
    
  }

  
  write.csv(mobileInfo, file= paste(name.brand,"_INFO.csv",sep=""), row.names = FALSE)
  
}