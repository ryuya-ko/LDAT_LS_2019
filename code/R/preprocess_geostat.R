setwd('/Users/mfjkou/sengokuLab/study/LDAT_LS_2019/')

install.packages("kokudosuuchi")
install.packages("FNN")
install.packages("sp")
install.packages("raster")
install.packages('sf', dependencies = TRUE) #sfをインストールしていなかった&うまく読み込めなかった

library(kokudosuuchi)
library(FNN)
library(sp)
library(raster)
library(sf)

# データ整備
## 鉄道駅データの整理
urls    <- getKSJURL("N02", fiscalyear=2008)                      # 鉄道駅データのURL取得 
station0<- getKSJData(urls$zipFileUrl, cache_dir = "cached_zip")  # 鉄道駅データを読み込み
station1<- translateKSJData(station0[[2]])                        # データの列名を入れる
station2<- as(station1, "Spatial")                                # sf→sp形式に変換
names(station2)
station3<- coordinates(station2)                                  # 鉄道駅を表す線の両端の緯度経度
# coordinatesで緯度経度取得
# 2重配列の中に長方形の緯度経度が入ったデータ[[i]][[1]]
station4<- matrix(0, nrow=length(station3), ncol=2)               # 各駅の緯度経度を入れていく空行列
for(i in 1:length(station3)){                             # 各駅について以下の処理を実行
  station4[i,]<-colMeans(station3[[i]][[1]])               # 緯度経度を線の両端の座標の平均値で与える
}

## 東京駅データの整理（東京駅距離は例では直線距離で近似）
tokyo_st<- cbind(139.767125, 35.681236)

##########################################
## 地価観測地点のデータ整理

urls  <- getKSJURL("L01", prefCode = 13, fiscalyear=2008)      # 地価公示データのURL
d0 <- getKSJData(urls$zipFileUrl, cache_dir = "cached_zip")    # 地価公示データの読込
d1 <-translateKSJData(d0[[1]])                                 # 列名の取得
d2 <-as(d1, "Spatial")                                         # sf→sp形式に変換
names(d2)
d3 <- d2[ d2$利用現況 %in% "住宅", ]                           # 住宅系用途の地価データのみ取出
d4 <- d3[ d3$標準地市区町村名称 %in% c("東大和","多摩","立川","日野"), ]#対象地域のデータのみ取出
coords <- coordinates(d4)                                      # データの緯度経度
d4$公示価格 <-as.numeric(as.character(d4$公示価格))  # なぜか因子になっている公示地価を数値に変換

station <- get.knnx(station4,coords,1)$nn.dist         # 駅距離の計算
# k-meansによる距離計算でクラスタ1に設定している->全ての点の中での最小距離にある点との距離を計算している
tokyo   <- get.knnx(tokyo_st,coords,1)$nn.dist         # 東京距離の計算
dat <- data.frame( coords, d4$公示価格, station, tokyo)# 地価データを整備
names(dat)<-c("px", "py", "price", "station", "tokyo") # 列名を定義

write.csv(dat,file = 'data/dev/house_price_raw.csv', row.names = FALSE)
##########################################
## 予測地点データ(ここではグリッドを仮定)を生成

urls  <- getKSJURL("N03", prefCode = 13, fiscalyear=2008)       # 観測データと同じ処理で
city0 <- getKSJData(urls$zipFileUrl, cache_dir = "cached_zip")  # 市区町村データ(shp)を取得
city1 <-translateKSJData(city0[[1]])
city2 <-as(city1, "Spatial")
names(city2)

city3 <-city2[ city2$市区町村名 %in% c("東大和市","多摩市","立川市","日野市"),]
ce <-extent(city3)                                              # 対象地域を囲む長方形
#rasterパッケージの関数 #拡張した長方形の座標を取得する
md0 <-as.data.frame(coordinates(raster(ce, ncol=30, nrow=42)))  # 長方形上でグリッドを生成
# raster objectを作成して緯度経度をとる
names(md0)      <-c("px","py")
coordinates(md0)<-c("px","py")
# sp objectを作成する
md0_over        <-over(md0, city3)                          # 各グリッドが対象地域上にあるか調べる
# overはx上の点でyに相当する地点があるか検索する関数
mcoords         <-coordinates(md0[is.na(md0_over$行政区域コード)==FALSE,])
# 対象地域上のグリッドのみ抽出

mstation <- get.knnx(station4,mcoords,1)$nn.dist            # 各グリッドからの駅距離
mtokyo   <- get.knnx(tokyo_st,mcoords,1)$nn.dist            # 各グリッドからの東京距離
mdat  <- data.frame( mcoords, NA, mstation, mtokyo)         # 予測地点データを整備
names(mdat)<-c("px", "py", "price", "station", "tokyo")     # 変数名を定義

write.csv(mdat, 'data/dev/house_price_pred.csv', row.names = FALSE)
