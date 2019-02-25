# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd

from geostat_pykrige import *

dat = pd.read_csv('/Users/LOng/sengokulab/LDAT_LS_2019/data/dev/house_price_raw.csv')
mdat = pd.read_csv('/Users/LOng/sengokulab/LDAT_LS_2019/data/dev/house_price_pred.csv')

print(dat.head(3))

latlon, indep = transform_data(dat, ['px', 'py'], ['station', 'tokyo'])
mlatlon, mindep = transform_data(mdat, ['px', 'py'], ['station', 'tokyo'])

model = LinearRegression(fit_intercept=False)

gmodel = RegressionKriging(regression_model=model,  variogram_model='exponential', nlags=15)

gmodel.fit(indep, latlon, y) #kriging学習

my = gmodel.predict(mindep, mlatlon)

#交差検証
pred_list = loo_cv(gmodel, dat, 'price', ['px', 'py'], ['station', 'tokyo'])

plt.scatter(y, pred_list)
plt.ylim = (11.8, 12.8)
plt.show()

mdat['price'] = my
mdat.head() #

gmdat = transform_to_gpd(mdat, 'px', 'py')
gmdat.head()

rang = set_rang(gmdat, 'price', 10)

res_fig = plot_kriging_res(gmdat, 'price', rang)

gmdat['sq_resid'] = gmodel.krige_residual(mlatlon)**2

rang = set_rang(gmdat, 'sq_resid', 10)
sq_resid_fig = plot_kriging_res(gmdat, 'sq_resid', rang)

res_fig.savefig('/Users/LOng/sengokulab/LDAT_LS_2019/data/dev/plot/kriging_res_plot.png')
sq_resid_fig.savefig('/Users/LOng/sengokulab/LDAT_LS_2019/data/dev/plot/kriging_resid_plot.png')
