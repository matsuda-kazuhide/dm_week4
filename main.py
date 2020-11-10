import datasets
X,Y = datasets.load_nonlinear_example1()
ex_X = datasets.polynomial2_features(X)
print(ex_X)
print(X[0])
print(Y)


import regression
model = regression.LinearRegression()
print(model.x)

import importlib
importlib.reload(regression)
model = regression.LinearRegression()
model.fit(ex_X,Y)
print(model.theta)

print(model.predict(ex_X))

print(model.score(ex_X,Y))
