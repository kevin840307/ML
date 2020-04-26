from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal as mvn
import numpy as np

from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


value_x = np.concatenate(((np.random.normal(0, 1, 400) + 1) * 100, (np.random.normal(0.5, 0.5, 400) + 1) * 200, (np.random.normal(0.1, 0.1, 400) + 1) * 400))
value_y = np.concatenate(((np.random.normal(0, 1, 400) + 1) * 20, (np.random.normal(0.5, 0.5, 400) + 1) * 100, (np.random.normal(0.1, 0.8, 400) + 1) * 10))
data = np.concatenate((np.reshape(value_x, (-1, 1)), np.reshape(value_y, (-1, 1))), axis=1)

kmeans = KMeans(n_clusters=3, max_iter=600, algorithm = 'auto')
fitted = kmeans.fit(data)
prediction = kmeans.predict(data)


plt.figure(figsize = (10,8))
def plot_kmeans(kmeans, X, n_clusters=3, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
        
plot_kmeans(kmeans, data)


plt.figure(figsize = (10,8))
def plot_scatters(X, n_clusters=3):
    # plot the input data
    ax = plt.gca()
    ax.axis('equal')
    labels = np.concatenate((np.random.uniform(0, 0, 400), np.random.uniform(1, 1, 400), np.random.uniform(2, 2, 400)))
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

        
plot_scatters(data)


class GMM():
    def __init__(self, C, times):
        self.C = C
        self.times = times

    def get_params(self):
        return (self.mu, self.pi, self.sigma)

    def calculate_mean_covariance(self, prediction):
        d = self.data.shape[1]
        labels = np.unique(prediction)
        self.initial_means = np.zeros((self.C, d))
        self.initial_cov = np.zeros((self.C, d, d))
        self.initial_pi = np.zeros(self.C)
        '''
        here, the first dimension of ids is 1, so ids[0] is the self.data size
        ids[0] = np.where(prediction == label) size

        1.使用kmean初始化全部類別
        2.pi預設，當下類別數量 / 全部類別數量
        3.means預設平均，依照每筆資料
        4.標準差計算協方差矩陣乘上pi
        '''
        index = 0
        for label in labels:
            ids = np.where(prediction == label)
            size = len(ids[0])
            self.initial_pi[index] = size / self.data.shape[0]
            self.initial_means[index,:] = np.mean(self.data[ids], axis=0)
            mean = self.data[ids] - self.initial_means[index,:]

            # initial_cov[index,:,:] = initial_pi[index] * np.cov(mean.T)
            self.initial_cov[index,:,:] = np.dot(self.initial_pi[index] * mean.T, mean) / size
            index += 1

        assert np.sum(self.initial_pi) == 1

        return (self.initial_means, self.initial_cov, self.initial_pi)

    def _initialise_parameters(self, data):
        self.data = data
        kmeans = KMeans(n_clusters=self.C, max_iter=500, algorithm = 'auto')
        fitted = kmeans.fit(self.data)
        prediction = kmeans.predict(self.data)
        self._initial_means, self._initial_cov, self._initial_pi = self.calculate_mean_covariance(prediction)
        
        return (self._initial_means, self._initial_cov, self._initial_pi)

    def _e_step(self):
        '''
        1.計算N筆資料在每一群的可能，也就是從高斯分部抽取
        2.除上每一群的可能，也就是機率
        '''
        N = self.data.shape[0]
        self.gamma = np.zeros((N, self.C))

        const_c = np.zeros(self.C)

        for c in range(self.C):
            self.gamma[:,c] = self.pi[c] * mvn.pdf(self.data, self.mu[c,:], self.sigma[c])

        gamma_norm = np.sum(self.gamma, axis=1)[:,np.newaxis]
        self.gamma /= gamma_norm

        return self.gamma

    def _m_step(self):
        '''
        1.偏微分推倒較複雜，用二維的推較好理解
        '''
        N = self.data.shape[0]
        C = self.gamma.shape[1]
        d = self.data.shape[1]

        self.pi = np.mean(self.gamma, axis=0)

        self.mu = np.dot(self.gamma.T, self.data) / np.sum(self.gamma, axis = 0)[:,np.newaxis]

        for c in range(C):
            x = self.data - self.mu[c,:]

            gamma_diag = np.diag(self.gamma[:,c])
            x_mu = np.matrix(x)
            gamma_diag = np.matrix(gamma_diag)

            sigma_c = np.matrix(x).T * gamma_diag * np.matrix(x)
            self.sigma[c,:,:] = (sigma_c) / np.sum(self.gamma, axis = 0)[:,np.newaxis][c]

        return self.pi, self.mu, self.sigma

    def _compute_loss_function(self):
        N = self.data.shape[0]
        C = self.gamma.shape[1]
        self.loss = np.zeros((N, C))

        for c in range(C):
            dist = mvn(self.mu[c], self.sigma[c],allow_singular=True)
            self.loss[:,c] = self.gamma[:,c] * (np.log(self.pi[c] + 0.00001) + dist.logpdf(self.data) - np.log(self.gamma[:,c] + 0.000001))
        self.loss = np.sum(self.loss)

        return self.loss

    def fit(self, data):
        
        d = data.shape[1]
        self.mu, self.sigma, self.pi =  self._initialise_parameters(data)
        
        try:
            for run in range(self.times):  
                self.gamma  = self._e_step()
                self.pi, self.mu, self.sigma = self._m_step()
                loss = self._compute_loss_function()
                
                if run % 10 == 0:
                    print("Iteration: %d Loss: %0.6f" %(run, loss))

        
        except Exception as e:
            print(e)
            
        
        return self

    def predict(self, data):
        labels = np.zeros((data.shape[0], self.C))
        
        for c in range(self.C):
            labels [:,c] = self.pi[c] * mvn.pdf(data, self.mu[c,:], self.sigma[c])
        labels  = labels .argmax(1)
        return labels 
    
    def predict_proba(self, data):
        post_proba = np.zeros((data.shape[0], self.C))
        
        for c in range(self.C):
            # Posterior Distribution using Bayes Rule, try and vectorise
            post_proba[:,c] = self.pi[c] * mvn.pdf(data, self.mu[c,:], self.sigma[c])
    
        return post_proba


model = GMM(3, times  = 100)

fitted_values = model.fit(data)
predicted_values = model.predict(data)

# # compute centers as point of highest density of distribution
centers = np.zeros((3,2))
for i in range(model.C):
    density = mvn(cov=model.sigma[i], mean=model.mu[i]).logpdf(data)
    centers[i, :] = data[np.argmax(density)]
    
plt.figure(figsize = (10,8))
plt.scatter(data[:, 0], data[:, 1],c=predicted_values ,s=50, cmap='viridis', zorder=1)

plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.5, zorder=2);

w_factor = 0.2 / model.pi.max()
for pos, covar, w in zip(model.mu, model.sigma, model.pi):
    draw_ellipse(pos, covar, alpha = w)









from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type='full').fit(data)
prediction_gmm = gmm.predict(data)
probs = gmm.predict_proba(data)



centers = np.zeros((3,2))
for i in range(3):
    density = mvn(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(data)
    centers[i, :] = data[np.argmax(density)]

plt.figure(figsize = (10,8))
plt.scatter(data[:, 0], data[:, 1],c=prediction_gmm ,s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6)

w_factor = 0.2 / gmm.weights_.max()
for pos, covar, w in zip(gmm.means_, gmm.covariances_ , gmm.weights_ ):
    draw_ellipse(pos, covar, alpha = w)