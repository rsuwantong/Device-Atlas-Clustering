
library(data.table)

setwd("D:/Tapad_UC1/Mobile_Atlas/Device_clustering")

# Reading the csv file
df = fread("techname_matched2atlas.csv", header = TRUE, sep = ",")
df = as.data.frame(df)
head(df)


df = as.data.frame(df)
df$AGE = 2017+2/12-as.numeric(df$TIME_RELEASED)
head(df)
# Removing the outliers
df = df[as.numeric(df$PRICE_RELEASED) <= 1000 & as.numeric(df$AGE) <=6 & 
          as.numeric(df$CAMERA_PIXELS) <= 25 & as.numeric(df$DIAGONAL_SCREEN_SIZE) <= 10 ,]

# Data to be used
df.utile = df[,c("PRICE_RELEASED", "AGE", "CAMERA_PIXELS")]
df.utile <- apply(df.utile,2,as.numeric)
df.scale = data.frame(scale(df.utile))

#load("OnlineRetailNbClust.RData")
# The above .Rdata file contains the results of NbClust (very long execution time), the script is below

# Code for the number of cluster to use (very long execution time)
library(NbClust) 
nc.km.all = NbClust(df.scale, min.nc=2, max.nc=20, method="kmeans", index = "all")
nc.hclust.all = NbClust(df.scale, min.nc=2, max.nc=20, method="ward.D2", index = "all")

# Plotting the results
library(factoextra)
fviz_nbclust(nc.km.all) + theme_minimal()
fviz_nbclust(nc.hclust.all) + theme_minimal()
