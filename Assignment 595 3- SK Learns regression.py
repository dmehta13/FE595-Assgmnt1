
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

linear_reg = LinearRegression()

bt = load_boston()

bt_price = bt.target

min_coef = 10000
max_coef = -10000
min_pos = 0
max_pos = 0


ystd = np.std(bt_price)
ymean = np.mean(bt_price)

N = len(bt_price)

Fac1 = bt_price - ymean
Fac1 = Fac1/ystd

Ninv = 1/(N-1)


for j in range(0,13,1):
    
    if j !=3:

      
        bt_data = bt.data[:, np.newaxis, j]
    
        bt_regresion = linear_reg.fit(bt_data,bt_price)
    
        bt_R2 = linear_reg.score(bt_data,bt_price)

        
        xstd = np.std(bt_data)
        xmean = np.mean(bt_data)
    
        Factor0 = bt_data - xmean
        Factor0 = Factor0/xstd
        bt_score = 0
        
        for h in range(1,N,1):
            bt_score = bt_score+(Factor0[h]*Fac1[h])
        bt_score = (bt_score*Ninv)
        
    
        if bt_R2 > max_coef:
            max_coef = bt_R2
            max_pos = j
        if bt_R2 < min_coef:
            min_coef = bt_R2
            min_pos = j

#printing the results        
print("The less influence predictor is " ,bt.feature_names[min_pos]," with R2 = ",min_coef)
print("The most influence predictor is " ,bt.feature_names[max_pos]," with R2 = ",max_coef)