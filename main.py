import datasets
X,Y = datasets.load_nonlinear_example1()
ex_X = datasets.polynomial3_features(X)
print(ex_X)
print(X[0])
print(Y)


import regression
model = regression.RidgeRegression(alpha=0.5)
model = regression.RidgeRegression()
print(model.alpha)

import importlib
importlib.reload(regression)
model = regression.LinearRegression()
model.fit(ex_X,Y)
print(model.theta)

print(model.predict(ex_X))

print(model.score(ex_X,Y))
