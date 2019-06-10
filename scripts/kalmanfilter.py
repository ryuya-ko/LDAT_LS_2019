import numpy as np


def kalman(y, x, SigP, SigX, SigY, F):
    '''
    カルマンフィルタ
    input:時点tの観測データ, 時点t-1の状態変数,
        分散共分散行列(状態変数, 観測・状態モデル), 状態変数と観測値の関係を表す行列
    output: 観測データの予測値, 状態変数のfiltering値, filtering分布の分散, 周辺尤度, 予測尤度の分散
    '''

    # 状態変数のt時点での分散
    SigP0 = SigP + SigX

    # 予測尤度の平均と分散
    # y0_til = np.dot(F, x)
    SX_til = np.dot(np.dot(F, SigP0), F.T) + SigY
    SX_til_inv = np.linalg.inv(SX_til)

    # カルマン利得
    K = np.dot(np.dot(SigP0, F.T), SX_til_inv)

    # 状態更新
    # 平均の算出
    y1_til = y - np.dot(F, x)
    x1 = x + np.dot(K, y1_til)
    # 分散の算出
    dim = F.shape
    if type(F) == int:
        dim = [1]
    In = np.identity(dim[1])
    Sig = In - np.dot(K, F)
    SigP1 = np.dot(Sig, SigP0)

    # 予測値の算出
    pred = np.dot(F, x1)

    # 事後分布の周辺尤度の計算
    det = np.linalg.det(SX_til)
    var = np.dot(np.dot(y1_til.T, SX_til_inv), y1_til)
    logz = 0.5*(np.log(det) - var)

    return pred, x1, SigP1, logz, SX_til
