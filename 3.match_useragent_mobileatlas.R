####################################################################################
# Name: Match useragent to mobile atlas
# Description: Matching phone technical names extracted from user agents to mobile atlas table 
# Version:
#   2016/10/04 RS: Initial version
#   2016/10/13 RS: Eliminate Apple and unuseful brands from the mobile atlas table
####################################################################################

# install.packages("data.table")
# install.packages("XML")
# install.packages("RCurl")
# install.packages("bitops")
# install.packages("pacman")
# .libPaths()
library(data.table)
library(dplyr)
library(rvest)
library(xml2)
library(httr)
library(XML)
library(bitops)
library(RCurl)

# Clear objects
rm(list=ls())

# Set default directory
default_directory <- "D:/Tapad_UC1/Mobile_Atlas/Device_Clustering/RESULTS"
#default_directory <- "R:/ShareData/Rata/Mobile_Atlas/RESULTS"

setwd(default_directory)


# set function
trim <- function (x) gsub("^\\s+|\\s+$", "", x)

getGoogleURL <- function(search.term, domain = '.co.th', quotes=TRUE) 
{
  search.term <- gsub(' ', '%20', search.term)
  if(quotes) search.term <- paste('%22', search.term, '%22', sep='') 
  getGoogleURL <- paste('http://www.google', domain, '/search?q=',
                        search.term, sep='')
}

getGoogleLinks <- function(google.url) {
  doc <- getURL(google.url, httpheader = c("User-Agent" = "R
                                           (2.10.0)"))
  html <- htmlTreeParse(doc, useInternalNodes = TRUE, error=function
                        (...){})
  nodes <- getNodeSet(html, "//h3[@class='r']//a")
  return(sapply(nodes, function(x) x <- xmlAttrs(x)[["href"]]))
}

#This function is to cut the alphabet at the end of Samsung model's name, e.g. sm-j700f will be transformed to sm-j700

cutAlphabetSamsung <- function(model.samsung){
  element <- gsub('.*sch-','',gsub('.*shv-','',gsub('.*sgh-','',gsub('.*gt-','',gsub('.*sm-','',model.samsung)))))
  root <- gsub('-.*','',model.samsung)
  dummy <- strsplit(element,split='')
  if (length(grep("[0-9]",dummy))>0){
    last.number <- tail(grep("[0-9]",dummy[[1]]),1)
    v <- dummy[[1]][1:last.number]
    v <- paste(v,collapse="")
    model.samsung.reduced <- paste(root,'-',v,sep="")
  } else model.samsung.reduced <- model.samsung
  return (model.samsung.reduced)
  
}



# start



dta <- fread("D:/Tapad_UC1/Mobile_Atlas/Device_clustering/devicelist_idsync_jan17.csv", integer64 = "character",header=TRUE)
dta <- data.frame(dta)

dta <- dta[!grepl('iphone|ipad',dta$MODEL),]

dta$MODEL <- sub('/.*','',dta$MODEL)
dta$MODEL <- sub('applewebkit.*','',dta$MODEL)
dta$MODEL <- gsub('_',' ',dta$MODEL)
dta$MODEL <- gsub('lg-','lg',dta$MODEL)
dta$MODEL <- gsub(')','',dta$MODEL)
dta$MODEL <- sub('\\(.*','',dta$MODEL)
dta$MODEL <- trim(as.character(dta$MODEL))

dta <- dta[!is.na(dta$MODEL),]
dta <- data.frame(dta)

dta[grepl('sm|gt',dta$MODEL) & !grepl('smart',dta$MODEL),1] <- sapply(dta[grepl('sm|gt',dta$MODEL) & !grepl('smart',dta$MODEL),1], function(x) cutAlphabetSamsung(x))


dta[,c(4:9)] <- lapply(dta[,c(4:9)], function(x) as.numeric(as.character(x)))



dta <- left_join(dta[!duplicated(dta$MODEL),c(1:8)], dta %>% group_by(MODEL) %>% summarise(REACH = sum(REACH)), by = c("MODEL" = "MODEL"))
nrow(dta)


temp = list.files(pattern="INFO.csv")
temp <- temp[!grepl('alcatel|allview|amazon|amoi|apple|archos|at&t|benefon|benq_siemens|bird|bosch|bq|casio|cat|chea|emporia|eten|fujitsu_siemens|ericsson|garmin_asus|haier|i_mate|innostream|inq|jolla|karbonn|kyocera|maxon|mitac|mitsubishi|modu|mwg|nec|neonode|nvidia|o2|orange |palm|parla|qtek|sagem|sendo|sewon|siemens|sonim|sony_ericsson|thuraya|t_mobile|tel_me|telit|thuraya|vertu|vk_mobile|vodafone|wnd|xcute',tolower(temp))]
brandlist <- tolower(sub('_INFO.*','',temp))
brandlist[brandlist=="i_mobile"] <- "i-mobile"



# write a table of the information of all brands

for (k in 1:length(temp)){
  info <- fread(temp[k], integer64 = "character",header=TRUE)
  
  info <- data.frame(info, stringsAsFactors = FALSE)
  info <- info[info$SCREEN_GSM > 3,]
  info <- unique(info)
  info$DETAIL_NAME_GSM <- tolower(info$DETAIL_NAME_GSM)
  info$MODEL_GSM <- tolower(info$MODEL_GSM)
  info$BRAND_GSM <- tolower(info$BRAND_GSM)
  
  if(brandlist[k]=="samsung"){ #for Samsung, "gear" is eliminated
    colnames(info) <-colnames(info.All)
    info<-info[!grepl("gear",info$MODEL_GSM),]
  }
  
  if (k==1){
    info.All <- info
  } else {
    info.All <- rbind(info.All,info)
  }
}



info.All <- cbind(info.All,gsub(',.*','',info.All$RELEASE_TIME_GSM),trim(gsub('.*,','',info.All$RELEASE_TIME_GSM)))
colnames(info.All)[7:8] <- c("RELEASE_YEAR","RELEASE_MONTH")
info.All$RELEASE_MONTH <-  gsub('[0-9]',NA,info.All$RELEASE_MONTH)

info.All[,c(7:8)] <-  lapply(info.All[,c(7:8)], function(x) as.character(x))

sapply(info.All, class)

info.All$RELEASE_MONTH[info.All$RELEASE_MONTH=='January'] <- 1
info.All$RELEASE_MONTH[info.All$RELEASE_MONTH=='February'] <- 2
info.All$RELEASE_MONTH[info.All$RELEASE_MONTH=='March'] <- 3
info.All$RELEASE_MONTH[info.All$RELEASE_MONTH=='April'] <- 4
info.All$RELEASE_MONTH[info.All$RELEASE_MONTH=='May'] <- 5
info.All$RELEASE_MONTH[info.All$RELEASE_MONTH=='June'] <- 6
info.All$RELEASE_MONTH[info.All$RELEASE_MONTH=='July'] <- 7
info.All$RELEASE_MONTH[info.All$RELEASE_MONTH=='August'] <- 8
info.All$RELEASE_MONTH[info.All$RELEASE_MONTH=='September'] <- 9
info.All$RELEASE_MONTH[info.All$RELEASE_MONTH=='October'] <- 10
info.All$RELEASE_MONTH[info.All$RELEASE_MONTH=='November'] <- 11
info.All$RELEASE_MONTH[info.All$RELEASE_MONTH=='December'] <- 12

info.All[,c(4:5,7:8)] <- lapply(info.All[,c(4:5,7:8)] , function(x) as.numeric(as.character(x)))
info.All$RELEASE_TIME <- info.All$RELEASE_TIME_GSM

info.All[,9] <- lapply(info.All[,9] , function(x) as.numeric(as.character(x)))
sapply(info.All, class)


info.All$RELEASE_TIME <- info.All$RELEASE_YEAR+(info.All$RELEASE_MONTH-1)/12

info.All <- info.All[info.All$RELEASE_YEAR>2006,]
info.All <- info.All[,c(-3)]
info.All <- cbind(info.All[,c(1:2,6:8)],info.All[,c(3:5)])
info.All <-info.All[order(info.All$RELEASE_YEAR,decreasing = FALSE),]
addinfo.All <- data.frame(matrix(NA, nrow = nrow(dta), ncol = 5))
colnames(addinfo.All) <- c("YEAR","MONTH","TIME","PRICE","SCREEN")




model_device.augmented<- data.frame(matrix(NA, nrow = nrow(dta), ncol = (ncol(info.All)+1)))
model_device.augmented[,1] <- dta[,c(1,3)]
colnames(model_device.augmented) <- c("MODEL","VENDOR","MODEL_GSM", "YEAR_RELEASED_GSM",   "MONTH_RELEASED_GSM", 
                                      "TIME_RELEASED_GSM",    "PRICE_RELEASED_GSM",  "DIAGONAL_SCREEN_SIZE_GSM" ,"DETAIL_NAME_GSM")



for (k in (1:nrow(dta))){

#for (k in (1:10)){  
  tryCatch({
    print(k)
    model_device <- dta$MODEL[k]
    info_brand <- info.All
    brand <- dta$VENDOR[k]
    
      
      for (m in 1:length(brandlist)){
        if (length(grep(brandlist[m],dta$VENDOR[k]))>0){
          info_brand <- info.All[info.All$BRAND_GSM==brandlist[m],]
          brand <- brandlist[m]
          print(brand)
        } 
      }
    
    
    
    
    info_brand <- info_brand[order(info_brand$RELEASE_YEAR,decreasing=FALSE),]
    info_brand <- info_brand[!is.na(info_brand$BRAND),]
    
    for (j in 1:nrow(info_brand)){
      
      
      if (length(grep(model_device,info_brand[j,c(2,8)][which( !is.na(info_brand[j,c(2,8)]), arr.ind=TRUE)]))>0) {
        
        print(j)
        addinfo.All[k,] <- info_brand[j,3:7]
        model_device.augmented[k,] <- c(model_device,brand, info_brand[j,2:7])
        print(model_device.augmented[k,] )
        
      }
    }
    
    # For Samsung whose gsmarena info does not include gt/ sm, e.g. Samsung Galaxy Note II N7100 (http://www.gsmarena.com/samsung_galaxy_note_ii_n7100-4854.php)
    # we cut the sm, gt so that it could find the matched expression in gsmarena tables
    
    if (is.na(addinfo.All[k,1]) & brand=='samsung' ){
      
      model_device_cut <- gsub('gt-','',gsub('sm-','',model_device))
      samsung_number <- gsub('[^0-9]','',model_device_cut) 
      samsung_alphabet <- gsub('[^a-z]','',model_device_cut) # n = note, t = tab
      
      if(samsung_alphabet=="n"){
        info_brand <- info_brand[grep("note",info_brand$MODEL),]
      } else if (samsung_alphabet=="t"){
        info_brand <- info_brand[grep("tab",info_brand$MODEL),]
      } else if (samsung_alphabet=="a"){
        info_brand <- info_brand[grepl("galaxy a",info_brand$MODEL)&!grepl("galaxy ace",info_brand$MODEL),]
      } else if (samsung_alphabet=="j"){
        info_brand <- info_brand[grepl("galaxy j",info_brand$MODEL),]
      }  
      
      for (j in 1:nrow(info_brand)){
        
        if (length(grep(model_device_cut,info_brand[j,c(2,8)][which( !is.na(info_brand[j,c(2,8)]), arr.ind=TRUE)]))>0) {
          
          print(j)
          addinfo.All[k,] <- info_brand[j,3:7]
          model_device.augmented[k,] <- c(model_device,info_brand[j,])
          print(model_device.augmented[k,] )
          
        }
      }
      
      if(is.na(addinfo.All[k,1])&(samsung_alphabet=="a"|samsung_alphabet=="j")){
        
        model_device_cut <- paste(samsung_alphabet,unlist(strsplit(samsung_number,split=''))[1],sep="")
        for (j in 1:nrow(info_brand)){
          
          if (length(grep(model_device_cut,info_brand[j,c(2,8)][which( !is.na(info_brand[j,c(2,8)]), arr.ind=TRUE)]))>0) {
            
            print(j)
            addinfo.All[k,] <- info_brand[j,3:7]
            model_device.augmented[k,] <- c(model_device,info_brand[j,])
            print(model_device.augmented[k,] )
            
          }
        }
        
        
      }
      
      
      # For Samsung phone that have not still found the match, we cut all the alphabet. In fact, there exists a case where the information on the gsmarena page
      # does not include the alphabet at all, e.g. as http://www.gsmarena.com/samsung_galaxy_note_8_0-5252.php has only the keyword 510 available
      # and we keep only the first 3 numbers.
      if(is.na(addinfo.All[k,1])){
        
        model_device_cut <- gsub('[^0-9]','',model_device_cut)
        e<- strsplit(model_device_cut,split='')
        v <- e[[1]][1:3]
        model_device_cut <- paste(v,collapse="")
        for (j in 1:nrow(info_brand)){
          
          if (length(grep(model_device_cut,info_brand[j,c(2,8)][which( !is.na(info_brand[j,c(2,8)]), arr.ind=TRUE)]))>0) {
            
            print(j)
            addinfo.All[k,] <- info_brand[j,3:7]
            model_device.augmented[k,] <- c(model_device,info_brand[j,])
            print(model_device.augmented[k,] )
            
          }
        }
      }
      
    } else if (is.na(addinfo.All[k,1])){ 
      # for other models whose still have not matched to the expressions in the gsmarena table yet, 
      # we cut the brand name from the model_device regex from kaidee.
      
      if (!brand=="unknown"){
        model_device_cut <- gsub(brand,'',model_device)
        model_device_cut <- trim(model_device_cut)
      } else 
        model_device_cut <- model_device 
      
      if (brand=="lenovo"){
        
        e<- strsplit(model_device_cut,split='')
        v <- e[[1]][1:3]
        model_device_cut <- paste(v,collapse="")
      }  
      
      for (j in 1:nrow(info_brand)){
        
        if (length(grep(model_device_cut,info_brand[j,c(2,8)][which( !is.na(info_brand[j,c(2,8)]), arr.ind=TRUE)]))>0) {
          
          addinfo.All[k,] <- info_brand[j,3:7]
          model_device.augmented[k,] <- c(model_device,info_brand[j,])
          print(model_device.augmented[k,] )
          
        }
      }
      
      
      
    }
    
    if (is.na(addinfo.All[k,1])){ 

      search.term <- paste(model_device,'+gsmarena',sep='')
      quotes <- "FALSE"
      search.url <- getGoogleURL(search.term=search.term, quotes=quotes)
      
      links <- getGoogleLinks(search.url)
      
      link <- links[1]
      link <- gsub('&sa.*','',gsub('.*q=','',link))
      brand <- gsub('_.*','',gsub('.*.com/','',link))
      info_brand <- info.All[info.All$BRAND==brand,]
      info_brand <- info_brand[!is.na(info_brand$BRAND),]
      model_device_cut <- gsub('-.*','',gsub('.*.com/','',link))
      model_device_cut <-  gsub(paste(brand,'_',sep=''),'',model_device_cut)
      model_device_cut <- gsub('_',' ',sub('_0','',model_device_cut))
   
      if (length(grep('galaxy mega',model_device_cut)>0)){
        samsung_number <- gsub('[^0-9]','',model_device_cut) 
        dummy <- strsplit(samsung_number,'')
        firstnum <- unlist(dummy)[1]
        model_device_cut <- paste(gsub('[0-9].*','',model_device_cut),firstnum,sep='') 
      }
      print(model_device_cut)
      for (j in 1:nrow(info_brand)){
        
        if (length(grep(model_device_cut,info_brand[j,c(2,8)][which( !is.na(info_brand[j,c(2,8)]), arr.ind=TRUE)]))>0) {
          print(j)
          addinfo.All[k,] <- info_brand[j,3:7]
          model_device.augmented[k,] <- c(model_device,info_brand[j,])
          print(model_device.augmented[k,] )
          
        }
      }
      
      
    }
    
    
    
  },error=function(e){}) 

  }



write.csv(model_device.augmented[!is.na(model_device.augmented$VENDOR),], file="mobile_atlas_gsmarena.csv", row.names = FALSE)