# リーディングDAT LSの地球統計モデリングの章の実装
setwd('~/sengokuLab/LDAT_LS_2019/')

# install.packages("gstat")
# install.packages("spBayes")
# install.packages("RColorBrewer")
library(sp)
library(gstat)
library(spBayes)
library(RColorBrewer)

## 前処理したデータを読み込む
dat <- read.csv('/Users/LOng/sengokulab/LDAT_LS_2019/data/dev/house_price_raw.csv')
mdat <- read.csv('/Users/LOng/sengokulab/LDAT_LS_2019/data/dev/house_price_pred.csv')

dat[1:5,]
coordinates(dat) =~ px+py

mdat[1:5,]
coordinates(mdat)=~px+py
## 分布の確認
### 観測データを地理情報付きデータに変換

display.brewer.all()
nc   <- 8
cols <- rev(brewer.pal(n = nc, name = "RdYlBu"))
cuts <- c(-Inf, quantile(dat$price,probs=seq(0.1,0.9,0.1)), Inf)
spplot(dat, "price",cuts=cuts, col.regions = cols, col="transparent",cex=1.5) # 住宅地価をプロット

## バリオグラムの推定
varioC <- variogram(object=log(price)~tokyo+station,data=dat,cloud=T)
plot(varioC)                                # バリオグラム雲

vario <- variogram(object=log(price)~tokyo+station,data=dat)
plot(vario)                                 # 経験バリオグラム

mvario <- fit.variogram(object=vario, model=vgm(psill=0.04,model="Exp",range=0.04,nugget=0.01))
# 理論バリオグラムの推定(vgm関数の中に入れる各値はパラメータ推定の初期値)
plot(vario,mvario)                          # 理論バリオグラム

#automapについては別途行う(アップされたコードに記載なし)


## Regression kriging
gmodel <- gstat(formula= log(price)~tokyo+station,data=dat, model=mvario) # モデルを定義
mpred  <- predict(gmodel,newdata=mdat)             # Regression kriging
# geostat関数でkriging, 地理情報付きデータで予測値と期待二乗誤差が返る
mdat@data$predRK<-exp(mpred$var1.pred)             # 予測値(対数値を元に戻す)
mdat@data$varRK <-mpred$var1.var                   # 期待二乗誤差

## Linear regression（比較用）
lmodel <-lm(log(price)~tokyo+station,data=dat@data)# 回帰
lmpred <-exp(predict(lmodel, newdata=mdat@data))
mdat@data$predLM<-lmpred                           # 予測値

## 可視化
nc  <- 10                                         # 10個に色分け
cols <- rev(brewer.pal(n = nc, name = "RdYlBu"))   # 色分け方法を指定
rang <-range(mdat@data$predRK)                     # 予測値の最小値・最大値
cuts<-seq(rang[1],rang[2],len=nc-1)                # 色の区切り値を指定

spplot(mdat, "predRK",cuts=cuts, col.regions = cols, col="transparent",cex=1.9, pch=15) # krigingの結果
# 隙間ができる場合はサイズcexを大きくする

spplot(mdat, "predLM",cuts=cuts, col.regions = cols, col="transparent",cex=1.9,pch=15) # 線形回帰の結果
#hotspotが消失してしまっている

#krigingによる期待二乗誤差をプロット
rang <-range(mdat@data$varRK)                      # 予測値の最小値・最大値
cuts<-quantile(mdat@data$varRK,probs=seq(0,1,0.1)) # 色の区切り値を指定
spplot(mdat, "varRK",cuts=cuts, col.regions = cols, col="transparent",cex=2,pch=15)

## Leave-one-out cross validatio
### Regression kriging
predRK <-krige.cv(log(price)~tokyo+station,dat, mvario)
# gstatの関数, デフォルトでLOO
### Linear regression
predLM <-NULL
for(i in 1:(dim(dat)[1])){
  dsub  <-dat@data[-i,] # 選んだ項だけ除外できる
  msub  <-dat@data[ i,]
  lmod  <-lm(log(price)~tokyo+station,data=dsub)
  predlm<-predict(lmod,  newdata=msub)
  predLM<-c(predLM, predlm)
}

### 結果比較
curve((x), 11.6, 12.8, xlab="", ylab="", xlim=c(11.6, 12.8), ylim=c(11.6, 12.8))
par(new=T)
plot(predRK@data[,c("observed","var1.pred")],pch=20, xlim=c(11.6, 12.8), ylim=c(11.6, 12.8)) # 真値vs予測値, y=x上にあればよい

plot(log(dat@data$price),predLM,pch=20) # krigingに比べてバラバラ

rmseLM <- sqrt(sum((predLM-log(dat@data$price))^2)/(dim(dat)[1]))
rmseLM
rmseRK <- sqrt(sum((predRK@data[,"observed"]-predRK@data[,"var1.pred"])^2)/(dim(dat)[1]))
rmseRK # RMSEでも半分以下

