### get the data of land price
library("kokudosuuchi")

urls <- getKSJURL("L01", prefCode = 8, fiscalyear=2010) # get from url
d0 <- getKSJData(urls$zipFileUrl, cache_dir = "cached_zip") # public land data
d1 <- translateKSJData(d0[[1]]) # change column names readable
d2 <- as(d1, "Spatial") # transform the format:sf->sp
names(d2)

d3 <- d2[ d2$利用現況 %in% "住宅", ] # pick up the housing land
coords <- coordinates(d3) # pick up the latlon

tokyo_st <- c(139.767125, 35.681236) #tokyo station
tokyo <- sqrt((tokyo_st[1]-coords[,1])^2+(tokyo_st[2]-coords[,2])^2) # distance from tokyo st

d3$駅からの距離 <-as.numeric(as.character(d3$駅からの距離))/1000 # change the support:meter->kilometer
d3$公示価格 <-as.numeric(as.character(d3$公示価格)) # convert from factor to numeric
dat <- data.frame( coords, d3$公示価格, d3$駅からの距離, tokyo) # pick up the essential part
names(dat)<-c("px", "py", "price", "station", "tokyo") # redefine the column names


source("http://aoki2.si.gunma-u.ac.jp/R/src/tolerance.R", encoding="euc-jp") # function for VIF test
#install.packages('spdep')
#install.packages('sphet')
library(spdep)
library(sphet)

dat[1:10,]
coords <- as.matrix(dat[,c("px", "py")]) # pick up latlon

Wnb <-knn2nb(knearneigh(coords, k=4))   # rook type spatial correlation
W <-nb2listw(Wnb, style="W")            # transform to matrix

x <- as.matrix(dat[,c("station", "tokyo")])
tolerance(x) #check the multicoliearity->unable on 10/Mar/2019

hist(dat$price)
hist(log(dat$price))
dat$ln_price<- log(dat$price)

### Analysis
## OLS
mod <- lm(ln_price~station+tokyo, dat)
mod
summary(mod)
lm.morantest(mod, W) #test for spatial correlation using the residuals

## selection of model
# Lagrange test
tres <- lm.LMtests(mod, W, test=c("LMerr", "LMlag"))
summary(tres)
# robust Lagrange test
tres <- lm.LMtests(mod, W, test=c("RLMerr", "RLMlag"))
summary(tres)

### models of spatial econometrics
## GMM estimation: fast, but not yet implemention of SDM, SDEM 
# SLM(Spatial Lag Model)
slm <- spreg(ln_price~station+tokyo,listw=W, dat, model="lag")
summary(slm)
# SEM(Spatial Error Model)
sem <- spreg(ln_price~station+tokyo,listw=W, dat, model="error")
summary(sem)
# SARAR model
sarar <- spreg(ln_price~station+tokyo,listw=W, dat, model="sarar")
summary(sarar)

## evaluation of direct and indirect effect
W2 <- as(W, "CsparseMatrix")
trMC <- trW(W2, type="MC")
sim_slm <- impacts(slm, R=200, tr=trMC)
summary(sim_slm, zstats=TRUE, short=TRUE) # need the argument 'evalues'. What's this?

sim_sarar<- impacts(sarar, R=200, tr=trMC)
summary(sim_sarar, zstats=TRUE, short=TRUE)