# ---------------------- BEFORE PolynomialFeatures ----------------------

import pandas as pd


df=pd.read_csv(r'c:\Users\dell\Desktop\udemy file\udemy file\08-Linear-Regression-Models\Advertising.csv')

df.head()

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

model=LinearRegression()

x=df.drop('sales',axis=1)
y=df['sales']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

model.fit(X_train,y_train)

y_hat=model.predict(X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error

mean_squared_error(y_test,y_hat)

mean_absolute_error(y_test,y_hat)

yy=y_test.mean()
xx=mean_absolute_error(y_test,y_hat)
print(xx/yy)

# r2
from sklearn.metrics import r2_score
print(r2_score(y_test,y_hat))

test_reduals=y_test - y_hat

import seaborn as sns

sns.scatterplot(x=y_test,y=test_reduals)



# ---------------------- AFTER PolynomialFeatures ----------------------

df=pd.read_csv(r'c:\Users\dell\Desktop\udemy file\udemy file\08-Linear-Regression-Models\Advertising.csv')

import pandas as pd

x=df.drop('sales',axis=1)
y=df['sales']

from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(degree=2,interaction_only=False)

poly_feture=poly.fit_transform(x)

from  sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(poly_feture, y, test_size=0.3, random_state=42)

from  sklearn.linear_model import LinearRegression
model=LinearRegression()

model.fit(X_train,y_train)

y_hat=model.predict(X_test)

model.coef_

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import numpy as np

r2_score(y_test,y_hat)

MAE=mean_absolute_error(y_test,y_hat)
MAE

msr=mean_squared_error(y_test,y_hat)
msr

rootmeanse=np.sqrt(msr)
rootmeanse

reduals=y_test - y_hat

import seaborn as sns


sns.scatterplot (x=y_test,y=reduals)

from joblib import dump,load
from sklearn.linear_model import LinearRegression
final_model2=LinearRegression()

final_model2.fit(poly_feture,y)

dump(final_model2,'model_liner_reg+polynomial')

model=load('model_liner_reg+polynomial')

