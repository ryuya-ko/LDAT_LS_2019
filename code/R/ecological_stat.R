# analysis of housing price in glasgow using ICAR--corresponding to text pp73-146
setwd('~/sengokulab/LDAT_LS_2019/code/R')
## install packages
#install.packages('CARBayes')
#install.packages('CARBayesdata')

## load library
library(CARBayes)
library(CARBayesdata)
library(RColorBrewer)
library(spdep)

data(GGHB.IG) # geo-data of glasgow
data(pricedata) # housing price data
pricedata$logprice<-log(pricedata$price) # log

dat <- merge(x=GGHB.IG, y=pricedata, by='IG', all.x=FALSE) # merge two data on column IG
dat <- spTransform(dat, CRS("+proj=longlat +datum=WGS84 +no_defs")) # define coordinate as latlon

nc <- 10 # 10colors
cuts <- seq(4,6,len=nc+1) # location of bins
cols <- rev(brewer.pal(n = nc, name = "RdYlBu"))  # which group
spplot(dat, "logprice",col.regions = cols,lwd=0.01,
       colorkey=list(at= cuts),at=cuts, col="transparent") # plot the price data

W.nb <- poly2nb(dat) # define weight matrix(intact=nearby)
W <- nb2mat(W.nb, style="B")  # transform to matrix form


### estimation of gaussian ICAR
### likelihood:mcmc, prior: ICAR or Leroux(normal dist)
## without explanatory variables
# Leroux prior distribution
resLeX <- S.CARleroux(formula=logprice ~ 1, data=pricedata,
                      family="gaussian", W=W, burnin=5000, n.sample=20000)
resLeX
# ICAR(special case of Leroux with rho=1)
resICAR <- S.CARleroux(formula=logprice ~ 1, data=pricedata,
                       family="gaussian", W=W, burnin=5000, n.sample=20000, rho=1)
resICAR

## with explanatory variable
# Leroux
resLeX2 <- S.CARleroux(formula=logprice ~ crime+rooms+driveshop+type, data=pricedata,
                       family="gaussian", W=W, burnin=5000, n.sample=20000)
resLeX2

# ICAR
resICAR2 <- S.CARleroux(formula=logprice ~ crime+rooms+driveshop+type, data=pricedata,
                        family="gaussian", W=W, burnin=5000, n.sample=20000, rho=1)
resICAR2

### visualization
## add the result to the data
dat$resLex <- resLeX$fitted.values
dat$resICAR <- resICAR$fitted.values
dat$resLex2 <- resLeX2$fitted.values
dat$resICAR2 <- resICAR2$fitted.values

# Lerox wo exp var
spplot(dat, "resLex",col.regions = cols,lwd=0.01,
       colorkey=list(at= cuts),at=cuts, col="transparent")

# ICAR w/o exp var
spplot(dat, "resICAR",col.regions = cols,lwd=0.01,
       colorkey=list(at= cuts),at=cuts, col="transparent")

# Leroux with exp
spplot(dat, "resLex2",col.regions = cols,lwd=0.01,
       colorkey=list(at= cuts),at=cuts, col="transparent")

# ICAR with exp
spplot(dat, "resICAR2",col.regions = cols,lwd=0.01,
       colorkey=list(at= cuts),at=cuts, col="transparent")

### GLM estimation using ICAR
### ICAR can be used to glm modeling since it does not distort distribution of data
### application to the analysis of the number of patient
data(GGHB.IG) # glasgow's shp
data(pollutionhealthdata) # number of patient in glasgow
dat   <- merge(x=GGHB.IG, y=pollutionhealthdata, by="IG", all.x=FALSE, duplicateGeoms = TRUE)
# merge data on 'IG'
head(dat@data)

## visualize the data
nc <- 10 # vasualize the data
cuts <- seq(0,220,len=nc+1)
cols <- rev(brewer.pal(n = nc, name = "RdYlBu"))
spplot(dat, "observed",col.regions = cols,lwd=0.01,
       colorkey=list(at= cuts),at=cuts, col="transparent")

## GLM est Poisson regression
# estimated desease map
resGLM<- S.glm(formula=observed ~ offset(log(expected))+pm10+jsa, data=dat@data, 
               family="poisson", burnin=5000,n.sample=20000, thin=20)
dat$resGLM<-resGLM$fitted.values
spplot(dat, "resGLM",col.regions = cols,lwd=0.01,
       colorkey=list(at= cuts),at=cuts, col="transparent") # no heat map

## GLM with ICAR
# define weighting matrix
W.nb <- poly2nb(dat, row.names =rownames(dat@data)) # declare the connection
W <- nb2mat(W.nb, style="B")                        # transform to the matrix

formula <- observed ~ offset(log(expected))+pm10+jsa
resICAR <- S.CARleroux(formula=formula, data=dat@data, family="poisson", W=W,rho=1,
                      burnin=5000,n.sample=20000) # result of the est(glm+ICAR)
resICAR

formula <- observed ~ offset(log(expected))+pm10+jsa
resLeX<- S.CARleroux(formula=formula, data=dat@data, family="poisson", W=W, 
                     burnin=5000,n.sample=20000) # GLM+Leroux
resLeX

formula <- observed ~ offset(log(expected))+pm10+jsa
resBym<- S.CARbym(formula=formula, data=dat@data, family="poisson", W=W, 
                  burnin=5000,n.sample=20000) # GLM+bym
resBym

## visualization
# histgram of coef of pm10(GLM vs GLM+Leroux)
hist(resGLM$samples$beta[,2],xlim=c(0,0.06),3,ylim=c(0,100),col="grey",border="grey", freq=FALSE)
hist(resLeX$samples$beta[,2],xlim=c(0,0.06),10,ylim=c(0,100),add=TRUE,border="dark green", freq=FALSE)

# histgram of coeff of poor rate(GLM vs GLM+Leroux)
hist(resGLM$samples$beta[,3],xlim=c(0.06,0.1),3,ylim=c(0,150),col="grey",border="grey", freq=FALSE)
hist(resLeX$samples$beta[,3],xlim=c(0.06,0.1),10,add=TRUE,border="dark green",ylim=c(0,150), freq=FALSE)

# histgram of estimated desease risk(first 20 areas)
boxplot(as.matrix(resGLM$samples$fitted[,1:20]),ylim=c(0,140))
boxplot(as.matrix(resLeX$samples$fitted[,1:20]),ylim=c(0,140))

dat@data$GLMfit <-resGLM$fitted.values
dat@data$ICARfit<-resICAR$fitted.values
dat@data$LeXfit <-resLeX$fitted.values
dat@data$BYMfit <-resBym$fitted.values

# observed number of patients
nc <- 10
cols <- rev(brewer.pal(n = nc, name = "RdYlBu"))
spplot(dat, "observed",col.regions = cols, col="transparent",
       colorkey=list(at= c(-Inf,seq(30,180,len=nc-1),Inf)),at=c(-Inf,seq(30,180,len=nc-1),Inf))

# estimated risk by glm
cols <- rev(brewer.pal(n = nc, name = "RdYlBu"))
spplot(dat, "GLMfit",col.regions = cols, col="transparent",
       colorkey=list(at= c(-Inf,seq(30,180,len=nc-1),Inf)),at=c(-Inf,seq(30,180,len=nc-1),Inf))

# est risk by glm ICAR
cols <- rev(brewer.pal(n = nc, name = "RdYlBu"))
spplot(dat, "ICARfit",col.regions = cols, col="transparent",
       colorkey=list(at= c(-Inf,seq(30,180,len=nc-1),Inf)),at=c(-Inf,seq(30,180,len=nc-1),Inf))

# Leroux
spplot(dat, "LeXfit",col.regions = cols, col="transparent",
       colorkey=list(at= c(-Inf,seq(30,180,len=nc-1),Inf)),at=c(-Inf,seq(30,180,len=nc-1),Inf))
# bym
spplot(dat, "BYMfit",col.regions = cols, col="transparent",
       colorkey=list(at= c(-Inf,seq(30,180,len=nc-1),Inf)),at=c(-Inf,seq(30,180,len=nc-1),Inf))

## include the group effect using INLA
#install.packages("INLA", repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)
library(INLA)
dat@data[1:5,]
dat@data$ID <- 1:length(dat@data[,1]) # individual ID for ICAR
dat@data$ID2<- dat@data$IG            # group ID for random effect

# GLM + ICAR + priors for each group(random effect)
formula <- observed ~ pm10 + jsa + f(ID, model = "bym", graph= W)  +  f(ID2, model="iid")
res <- inla( formula, offset = log(expected), family = "poisson", data = dat@data)
summary(res)


res$summary.fitted.values

