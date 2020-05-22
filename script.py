import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices

data = pd.read_csv("./data.csv", parse_dates=True, index_col=[0])

def run_regression(s, p):
    y, X = dmatrices(s, data, return_type='dataframe')
    n = len(y)
    poisson = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    mus = poisson.mu
    phi_hat = np.sum((y.values[:, 0] - mus)**2 / mus) / (n - p)
    nb2 = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=phi_hat, link=sm.families.links.log)).fit()
    return nb2

print(run_regression("Cases ~ Daily_Dev", 1).summary())
print(run_regression("Hospitalizations ~ Daily_Dev", 1).summary())
print(run_regression("Deaths ~ Daily_Dev", 1).summary())

print(run_regression("Deaths ~ Daily_Dev + Day_First_Death", 2).summary())

print(run_regression("Deaths ~ Daily_Dev + Day_First_Death + Schools_Closed*Day_First_Death + Non_Essential_Closed*Day_First_Death + Mask_Mandatory*Day_First_Death", 8).summary())