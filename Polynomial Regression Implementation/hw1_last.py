
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

training_data = np.loadtxt("train.csv", delimiter=",", dtype=np.float32)
testing_data = np.loadtxt("test.csv", delimiter= ",", dtype=np.float32)

x = training_data[:,0].reshape(-1,1)
y = training_data[:,1]

x_test = testing_data[:,0].reshape(-1,1)
y_test = testing_data[:,1]

errors_training = {}
errors_testing = {}

for degree in range(0,7):
    polynomial = PolynomialFeatures(degree=degree)
    x_poly = polynomial.fit_transform(x)

    reg = LinearRegression()
    reg.fit(x_poly, y)

    x_values = np.linspace(-2.2,2.2,100).reshape(-1,1)
    x_values_poly = polynomial.transform(x_values)

    y_values = reg.predict(x_values_poly)

    squared_error_training = np.sum((reg.predict(polynomial.transform(x)) - y)**2)
    squared_error_testing = np.sum((reg.predict(polynomial.transform(x_test)) - y_test)**2)

    print(squared_error_testing)
    print(squared_error_training)

    errors_training[degree] = squared_error_training
    errors_testing[degree] = squared_error_testing

    plt.scatter(x, y)
    plt.plot(x_values, y_values, color = "r")
    plt.show()
        
    plt.scatter(x_test, y_test)
    plt.plot(x_values, y_values, color = "r")
    plt.show()

x = errors_training.keys()
y = errors_training.values()

plt.scatter(x,y)
plt.plot(y)
plt.show()

x_test = errors_testing.keys()
y_test = errors_testing.values()

plt.scatter(x_test,y_test)
plt.plot(y_test)
plt.show()

