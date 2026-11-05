# %%
%load_ext autoreload
%autoreload 2

import iml  # helper functions and classes (see iml/)
import numpy as np  # array processing
import pandas as pd  # dataframes, reading files
import matplotlib.pyplot as plt  # making plots

# %% [markdown]
# **Group members:** Emin Cesme, Valentine Scillia, Paulina Pervukhina

# %% [markdown]
# ## Part 1: Partial Derivatives

# %% [markdown]
# ### (a)
# > Compute the derivative $\partial L / \partial x(i).$ Leaving all other
# > variables fixed, what value of $x(i)$ minimizes $L$?

# %% [markdown]
# $$
# \frac{\partial L}{\partial x(i)} = \sum_{j=1}^{N} \frac{\partial |\mu - x(j)|}{\partial x(i)} + \frac{\partial (\lambda \mu^2)}{\partial x(i)}
# $$
# let 
# $ f(x) = |x| $
# then 
# $ f'(x) = sign(x)$ where
# $$
# \operatorname{sign}(x) =
# \begin{cases}
#   1 & \text{if } x > 0 \\
#   -1 & \text{if } x < 0
# \end{cases}
# $$
# so
# $$
# \frac{\partial L}{\partial x(i)} = \sum_{j=1}^{N} \frac{\partial f(\mu - x(j))}{\partial x(i)} + 0
# $$
# the only time when the partial derivative of x(i) is non-zero for the sum terms is when j=i so
# $$
# \frac{\partial L}{\partial x(i)} = f'(\mu - x(i)) * (\mu - x(i))'
# $$
# $$
# = sign(\mu - x(i)) * (-1)
# $$
# and therefore, 
# $$
# \frac{\partial L}{\partial x(i)} = -sign(\mu - x(i))
# $$

# #### When is L minimal?
# For x(i) to minimize L, we need $\frac{\partial L}{\partial x(i)}$ to be equal to 0 or the derivative to change sign from before that point to after that point (from negative to positive) <br>
# When $ \mu = x(i) $ , $sign(\mu - x(i))$ is not defined but by drawing the function we can see that the derivative/slope before is negative and after is positive and that L (looks like a V) will be minimal there. 



# %% [markdown]
# ### (b)
# > Compute the derivative $\partial L / \partial \mu.$ If $\lambda = 0,$ what
# > values of $\mu$ minimize $L$?

# %% [markdown]
# $$
# \frac{\partial L}{\partial \mu} = \sum_{j=1}^{N} \frac{\partial |\mu - x(j)|}{\partial \mu} + \frac{\partial (\lambda \mu^2)}{\partial \mu}
# $$
# let 
# $ f(x) = |x| $
# then 
# $ f'(x) = sign(x)$ where
# $$
# \operatorname{sign}(x) =
# \begin{cases}
#   1 & \text{if } x > 0 \\
#   -1 & \text{if } x < 0
# \end{cases}
# $$
# so
# $$
# \frac{\partial L}{\partial \mu} = \sum_{j=1}^{N} \frac{\partial f(\mu - x(j))}{\partial \mu} + 2\lambda\mu
# $$
# $$
# \frac{\partial L}{\partial \mu} = \sum_{j=1}^{N} f'(\mu - x(j)) * (\mu - x(j))' + 2\lambda\mu
# $$
# $$
# = \sum_{j=1}^{N} sign(\mu - x(j)) * (1) + 2\lambda\mu
# $$
# and therefore, 
# $$
# \frac{\partial L}{\partial \mu} = \sum_{j=1}^{N} sign(\mu - x(j)) + 2\lambda\mu
# $$

# #### If $\lambda = 0,$ what values of $\mu$ minimize $L$?
# If $\lambda = 0$, the second term of $\frac{\partial L}{\partial \mu}$ is also 0 so the only significant term left is the sum. <br>
# To minimize a function, we want to find when the derivative is equal to 0 <br>
# The question becomes, what values of $\mu$ makes $\sum_{j=1}^{N} sign(\mu - x(j))$ as close to 0 as possible <br>
# The sum will be closest to 0 when $\mu$ has as many x(j) on the left and on the right <br>
# This means that the first terms of the sum will be 1 and the last will be -1. Ideally we have as many -1s as 1s and then the sum will be exactly 0 <br>
# There are two cases: <br>
# There is an odd amount of x(i): $\mu$ should just be the middle value of the set of x(j) <br>
# There is an even amount of x(i): $\mu$ can be any value between the two middle values of the set x(j) <br>



# %% [markdown]
# ### (c)
# > Compute the derivative $\partial L / \partial \theta.$

# %% [markdown]
# With the chain rule, we can rewrite the problem: 
# $$
# \frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial \mu}.\frac{\partial \mu}{\partial \theta}
# $$
# 
# We know from exercise b) that:
# $$
# \frac{\partial L}{\partial \mu} = \sum_{j=1}^{N} sign(\mu - x(j)) + 2\lambda\mu
# $$ 
# 
# and $ \frac{\partial \mu}{\partial \theta} = 3\theta^2 $ since $\mu = \theta^3 $
# 
# Therefore, 
# 
# $$
# \frac{\partial L}{\partial \theta} = (\sum_{j=1}^{N} sign(\theta^3 - x(j)) + 2\lambda\theta^3) . 3\theta^2
# $$ 
# 

# %% [markdown]
# ### (d)
# > Compute $$
# > \frac{\partial}{\partial z_i} \ln \operatorname{softmax}(z)_j.
# > $$
# > (Hint: first define $Z = \ln \sum_{l = 1}^k e^{z_l}$ and compute
# > $\partial Z / \partial z_j.$)

# %% [markdown]
# We know the chain rule for ln gives us this:
# $$
# \frac{\partial}{\partial x} \ln \operatorname{f}(x) = \frac{1}{\operatorname{f}(x)} . \operatorname{f'}(x)
# $$
# 
# First I define $Z = \ln \sum_{l = 1}^k e^{z_l}$ <br>
# Then
# $$
# \frac{\partial}{\partial z_i} Z = \frac{1}{\sum_{l=1}^k e^{z_l}} . \frac{\partial}{\partial z_i} \sum_{l=1}^k e^{z_l}
# $$
# The second term can be reduced to only $ e^{z_i} $ since all the other terms of the sum are constants <br>
# Therefore, we get 
# $$ 
# \frac{\partial}{\partial z_i} Z = \frac{e^{z_i}}{\sum_{l=1}^k e^{z_l}}
# $$ 
# This is exactly our $\operatorname{softmax}(z)_i$ so $ \frac{\partial}{\partial z_i} Z = \operatorname{softmax}(z)_i $
# <br><br>
# Now, we go back to our original function that we want to compute: $ \frac{\partial}{\partial z_i} \ln \operatorname{softmax}(z)_j$ 
# Using the ln rules we can rewrite this as 
# $$
# = \frac{\partial}{\partial z_i} \ln \frac{e^{z_j}}{\sum_{l=1}^k e^{z_l}} = \frac{\partial}{\partial z_i} \ln e^{z_j} - \frac{\partial}{\partial z_i} \ln \sum_{l=1}^k e^{z_l}
# $$
# The second term is exactly $\frac{\partial}{\partial z_i} Z $ which we already have. We replace it:
# $$
# \frac{\partial}{\partial z_i} \ln \operatorname{softmax}(z)_j = \frac{\partial}{\partial z_i} \ln e^{z_j} - \operatorname{softmax}(z)_i
# $$ 
# 
# Now we compute: 
# $$ 
# \frac{\partial}{\partial z_i} \ln e^{z_j} = \frac{\partial}{\partial z_i} z_j
# $$ 
# There are two cases, either j = i and then the result is 1. Otherwise it's 0.<br>
# Let 
# $$ 
# \operatorname{is_i}(x) =
# \begin{cases}
#   1 & \text{if } x = i \\
#   0 & \text{otherwise } 
# \end{cases}
# $$ 
# Then 
# $$
# \frac{\partial}{\partial z_i} \ln e^{z_j} = \operatorname{is_i}(j)
# $$
# <br>
# When we put everything together: 
# $$
# \frac{\partial}{\partial z_i} \ln \operatorname{softmax}(z)_j = \operatorname{is_i}(j) - \operatorname{softmax}(z)_i
# $$


# %% [markdown]
# ## Part 2: Linear Regression

# %% [markdown]
# ### (a)
# > Consider the linear model, defined by parameters $(\theta_0, \theta_1)$,
# > which predicts $$
# > \hat y(i) = \theta_0 + \theta_1 x(i)
# > $$ for each index
# > $i.$ Express the gradient $\nabla L$ of mean squared error as a function of
# > the parameter vector $\theta$ and the datapoints $(x(i), y(i)).$
# %% [markdown]
# The mean squared error is:
# $$L(\theta) = \frac{1}{n} \sum_{i=1}^{n} (\theta_0 + \theta_1 x(i) - y(i))^2$$
# First, the partial derivative w.r.t. $\theta_0$:
# $$\frac{\partial L}{\partial \theta_0} = \frac{2}{n} \sum_{i=1}^{n} (\theta_0 + \theta_1 x(i) - y(i))$$
# Next, the partial derivative w.r.t. $\theta_1$:
# $$\frac{\partial L}{\partial \theta_1} = \frac{2}{n} \sum_{i=1}^{n} (\theta_0 + \theta_1 x(i) - y(i))\,x(i)$$
# The gradient is:
# $$\nabla L = \begin{bmatrix} \frac{2}{n} \sum_{i=1}^{n} (\theta_0 + \theta_1 x(i) - y(i)) \\ \frac{2}{n} \sum_{i=1}^{n} (\theta_0 + \theta_1 x(i) - y(i))\,x(i) \end{bmatrix}$$

# %% [markdown]
# ### (b)
# > Extend the `SimpleRegressor` class from `iml/simple_regressor.py`,
# > implementing the method `fit(self, x, y, lr, steps)` to implement gradient
# > descent on mean squared error. Run `fit()` on the _unnormalized_ data
# > vectors `x` and `y`, and show the training process over 20 iterations with
# > `SimpleRegressor.show_training()`. Find a learning rate that makes the loss
# > decrease.

# %% 

class SimpleRegressor:
    def __init__(self):
        self.weights_ = np.zeros(2)  # a vector with the two parameters of our model

    def predict(self, x):
        return self.weights_[0] + x * self.weights_[1]

    def fit(self, x, y, lr: float, steps: int = 20) -> None:
        # x and y are vectors of the same length

        self.weights_ = np.zeros(2)  # a vector with the two parameters of our model

        # we keep copies of the x, y and the predicted outputs at each
        # iteration to use them in show_training() later
        self.x_train_ = x.copy()
        self.y_train_ = y.copy()
        self.y_predicted_ = np.zeros((steps + 1, y.shape[0]))
        self.y_predicted_[0] = self.predict(x)

        for i in range(steps):
            error = self.predict(x) - y
            gradient_theta_0 = 2 * error.mean()
            gradient_theta_1 = 2 * (error * x).mean()
            # here: update self.weights_?
            self.weights_[0] -= lr * gradient_theta_0
            self.weights_[1] -= lr * gradient_theta_1

            self.y_predicted_[i + 1] = self.predict(x)
            
    def show_training(self):
        show_regression_training_(self.x_train_, self.y_train_, self.y_predicted_)

# %%
x, y = iml.load_powerplant()
regressor = iml.SimpleRegressor()
regressor.fit(x, y, lr=0.000001, steps=20)
regressor.show_training()

# %%
loss = np.mean((regressor.y_predicted_ - y) ** 2, axis=1)
print("Initial loss:", loss[0])
print("Final loss:", loss[-1])

# %% [markdown]
# Initial loss: 204650.77317070705
# Final loss: 197836.66588333726
# The loss decreases slowly by every iteration. 
# The unnormalized data has a large range, so the gradient w.r.t $\theta_1$ is much larger than the one for $\theta_0$.

# %% [markdown]
# ### (c)
# > Now, normalize your data in some way and run `fit()` and `show_training()`
# > again, ensuring that you converge to an acceptable model in 20 iterations.

# %%
# Normalize the input data
x_norm = (x - np.mean(x)) / np.std(x)
y_norm = (y - np.mean(y)) / np.std(y)

regressor_norm = iml.SimpleRegressor()
regressor_norm.fit(x_norm, y_norm, lr=0.1, steps=20)

regressor_norm.show_training()

# %%
loss_norm = np.mean((regressor_norm.y_predicted_ - y_norm) ** 2, axis=1)

print("Initial loss:", loss_norm[0])
print("Final loss:", loss_norm[-1])
print("Loss decreases every step:", np.all(np.diff(loss_norm) < 0))

# %% [markdown]
# After normalizing, both parameters have gradients of similar scale, so a much larger learning rate is stable.
# The loss now decreases at every step and the line converges to the data in the 20 iterations.
# %% [markdown]
# ## Part 3: k-Means and PCA

# %% [markdown]
# ### (a)
# > Suppose we didn't have access to the species attribute of the iris dataset.
# > Implement the Lloyd-Forgy algorithm for $k$-means and use it to infer a
# > partition of the iris dataset into three species. Compute the confusion
# > matrix and the accuracy of the partition relative to the true species.

# %%
iris = iml.load_iris()

X = iris.data
y_true = iris.target

print("Data shape:", X.shape)
print("Number of species:", len(iris.target_names))
print("Species:", iris.target_names)

# %%
def k_means(X, k, max_iterations=100):
    # choose initial centroids randomly
    random_indices = np.random.choice(len(X), size=k, replace=False)
    centroids = X[random_indices].copy()
    for i in range(max_iterations):

        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # calculate distances

        labels = np.argmin(distances, axis=1)  # assign points to clusters

        # recalculate centroids
        new_centroids = np.array([
            X[labels == c].mean(axis=0) if np.any(labels == c) else centroids[c]
            for c in range(k)
        ])
        # if centroids stop changing -> stop
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

# %%
np.random.seed(42)

k = 3
y_pred, centroids = k_means(X, k)

print("Predicted cluster labels:")
print(y_pred)

print("Final centroids:")
print(centroids)

# %%
from sklearn.metrics import confusion_matrix, accuracy_score
from itertools import permutations

best_accuracy = 0

for permutation in permutations(range(3)):
    y_pred_mapped = np.array([permutation[label] for label in y_pred])
    accuracy = accuracy_score(y_true, y_pred_mapped)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_prediction = y_pred_mapped

cm = confusion_matrix(y_true, best_prediction)

print("Confusion matrix:")
print(cm)

print("Accuracy:", best_accuracy)
print("Accuracy (%):", best_accuracy * 100)

# %% [markdown]
# We implemented the Lloyd-Forgy k-means algorithm using Numpy.
# We initialized three random centroids and assigned each observation to its nearest centroid using Euclidean distance.  
# Then recalculated the centroids as the mean of the observations in each cluster.
# This process was repeated until the centroids stopped changing.
#
# Since k-means assigns arbitrary numbers to clusters, we matched the
# predicted cluster labels with the actual species labels before
# calculating the confusion matrix and accuracy.
#
# The reported accuracy was 89.33%. Setosa was perfectly separated, while some of the versicolor and virginica observations were assigned to the wrong cluster. 
# This shows that these two species are more difficult to distinguish using k-means.

# %% [markdown]
# ### (b)
# > Investigate `data/ulu.csv` using PCA and explain how this dataset was
# > produced. Using scikit-learn, run both $k$-means clustering and DBSCAN on
# > this dataset. Which method produces more reasonable "clusters"?



# %%
from sklearn.decomposition import PCA

data = iml.load_ulu()
print(data.shape)

pca = PCA(n_components=3)
z = pca.fit_transform(data)
print("explained variance ratio:", pca.explained_variance_ratio_.round(4))

fig, axs = plt.subplots(1, 2, figsize=(14, 4))
axs[0].scatter(z[:, 0], z[:, 1], s=1)
axs[0].set(title="Projection onto PC1 and PC2", xlabel="PC1", ylabel="PC2", aspect="equal")
axs[1].hist(z[:, 2], bins=60)
axs[1].set(title="Histogram of PC3", xlabel="PC3", ylabel="number of points")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(14, 4))
sc = ax.scatter(z[:, 0], z[:, 1], c=z[:, 2], s=2, cmap="coolwarm")
ax.set(title="Projection onto PC1 and PC2, coloured by PC3", xlabel="PC1", ylabel="PC2", aspect="equal")
plt.colorbar(sc, ax=ax, label="PC3")
plt.show()

# %%
from sklearn.cluster import KMeans, DBSCAN

kmeans_labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(data)
dbscan_labels = DBSCAN(eps=0.3, min_samples=10).fit_predict(data)
print("k-means cluster sizes:", np.bincount(kmeans_labels))
print("DBSCAN cluster sizes:", np.bincount(dbscan_labels[dbscan_labels >= 0]), "noise:", np.sum(dbscan_labels == -1))

fig, axs = plt.subplots(2, 1, figsize=(14, 8))
axs[0].scatter(z[:, 0], z[:, 1], c=kmeans_labels, s=2, cmap="tab10")
axs[0].set(title="k-means (k = 2)", xlabel="PC1", ylabel="PC2", aspect="equal")
axs[1].scatter(z[:, 0], z[:, 1], c=dbscan_labels, s=2, cmap="tab10")
axs[1].set(title="DBSCAN (eps = 0.3, min_samples = 10)", xlabel="PC1", ylabel="PC2", aspect="equal")
plt.tight_layout()
plt.show()

# %% [markdown]
# PCA. We applied PCA with 3 components. The first component explains 87.3% of the variance, the second 12.0% and the third only 0.7%. When we plot the first two components, the points fill a rectangle, so the data is almost flat, like a plate. But the histogram of the third component has two separate peaks, about 1 apart. This means the points are in two parallel layers. When we colour the PCA plot by the third component, the smaller layer shows the text uni.lu. How the data set was produced. We think the data was made from a black-and-white image of the text “uni.lu”. Every pixel became a point on a grid (the points are about 0.1 apart). The background points and the letter points were put at two different heights, about 1 apart. Then some noise was added and the whole thing was rotated in 3D, so the two layers do not line up with the original x, y, z axes. PCA turns it back: the first two components are the directions along the plate, and the third component is the direction between the two layers. Interestingly, the text is in the component with the smallest variance, so if we only kept the first two components we would lose the text.
# Clustering. We used k-means with k = 2, because there are two layers. But k-means just cuts the plate into a left half and a right half (4389 points each), and both halves contain background and letters. This is because k-means looks for round groups of points, and the plate is much longer than the distance between the layers. DBSCAN with eps = 0.3 and min_samples = 10 works much better. It finds one cluster for the background (6935 points) and one cluster for each separate part of the text (u, n, the stem of the i, the dot of the i, the period, l, u), with no noise points. It works because points in the same layer are close to each other (about 0.1 apart), but the two layers are about 1 apart, so with eps = 0.3 DBSCAN never connects the two layers. Also, we did not have to tell DBSCAN how many clusters to find.
# Conclusion: DBSCAN gives much more reasonable clusters than k-means.
