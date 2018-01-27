import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model
np.random.seed(0)


n_samples = 1000
n_outliers = 50


def get_polynomial_samples(n_samples=1000):
    X = np.array(range(1000)) / 100.0
    np.random.shuffle(X)

    coeff = np.random.rand(2,) * 3

    y = coeff[0]*X**2 + coeff[1]*X + 10
    X = X.reshape(-1, 1)
    return coeff, X, y


def add_square_feature(X):
    X = np.concatenate([(X**2).reshape(-1,1), X], axis=1)
    return X


coef, X, y = get_polynomial_samples(n_samples)

# Add outlier data
X[:n_outliers] = 10 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -10 + 10 * np.random.normal(size=n_outliers)

# Fit line using all data
lr = linear_model.LinearRegression()
lr.fit(add_square_feature(X), y)

# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(add_square_feature(X), y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(add_square_feature(line_X))
line_y_ransac = ransac.predict(add_square_feature(line_X))

# Compare estimated coefficients
print("Estimated coefficients (true, linear regression, RANSAC):")
print(coef, lr.coef_, ransac.estimator_.coef_)

lw = 2
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
            label='Outliers')
plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
         label='RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()
