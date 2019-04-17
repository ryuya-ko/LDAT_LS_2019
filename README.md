# LDAT_LS_2019
2019/2/14~15に統計数理研究所で開催された【リーディングDAT講座】L-S 地理情報と空間モデリングの実装など

# scripts
krigingを行うにあたって必要な関数群

## Regression Kriging
1. OLS推定
    - 残差を算出する
2. 残差からvariogramを推定する
    - バリオグラム雲を```get_diff```で計算
    - ```auto_vario```でバリオグラムを推定する
3. GLS推定
    - 共分散関数を推定する(上記バリオグラムを用いる)
        - 観測値間の距離行列作成```calc_distance_matrix```
            - sillと有効レンジを算出```calc_c0```
            - 共分散関数を推定```est_covariance_matrix```
        - 共分散関数を重みにしてGLS```do_gls```
4. Kriging
    - 予測データと観測データの共分散関数を推定する
        - 予測データと観測データの距離行列作成```calc_distance_new_data```
        - 共分散関数を推定```est_covariance_matrix```
    - glsによる予測値を空間補間
        - 予測値の算出```put_pred_val```
        - 空間補間```do_kriging```
 

## spatial-temporal kriging