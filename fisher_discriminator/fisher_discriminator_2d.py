import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def generate_data(num_sample, mu1, mu2, sigma1, sigma2):
    c1_data = np.random.multivariate_normal(mu1, sigma1, num_sample)
    c2_data = np.random.multivariate_normal(mu2, sigma2, num_sample)
    return c1_data, c2_data

class linear_regressor():
    def __init__(self, c1_data, c2_data):
        self.c1_data = np.zeros((c1_data.shape[0], 4))
        self.c1_label = -1 * np.ones(c1_data.shape[0])
        self.c2_data = np.zeros((c2_data.shape[0], 4))
        self.c2_label = np.ones(c2_data.shape[0])
        
        # first column is all 1s
        # last column is -1 for c1, and +1 for c2
        self.c1_data[:,0] = 1
        self.c1_data[:,1:3] = c1_data
        self.c1_data[:,3] = -1
        self.c2_data[:,0] = 1
        self.c2_data[:,1:3] = c2_data
        self.c2_data[:,3] = 1

        self.X = np.concatenate((self.c1_data, self.c2_data))
        self.y = np.concatenate((self.c1_label, self.c2_label))

    def train(self, reg=0.01):
        # zero-center x
        X = self.X
        y = self.y
        # B = (Xt X)^-1 Xt y
        Xt_X_inv = np.linalg.inv(np.matmul(X.T, X) + np.eye(4)*reg)
        self.B = np.dot(np.matmul(Xt_X_inv, X.T), y)
        return self.B

class fisher_discriminator():
    def __init__(self, c1_data, c2_data):
        self.c1_data = c1_data
        self.c2_data = c2_data
        self.mu1, self.mu2 = self.calc_means()
        self.cov1, self.cov2 = self.calc_covariances()

    def train(self,):
        # (sig1 + sig2)^-1
        s = np.linalg.inv(self.cov1 + self.cov2)
        dmu = self.mu1 - self.mu2
        u = np.matmul(s, dmu)
        u_norm = u / np.sqrt(np.dot(u.T, u)) # normalize 
        return u_norm

    def calc_means(self,):
        mu1 = np.sum(self.c1_data, axis=0)
        mu1 /= self.c1_data.shape[0]
        mu2 = np.sum(self.c2_data, axis=0)
        mu2 /= self.c2_data.shape[0]
        return mu1, mu2

    def calc_covariances(self,):
        # cov = X^T X, where x is centered at 0
        c1 = self.c1_data - self.mu1
        c2 = self.c2_data - self.mu2
        cov1 = np.matmul(c1.T, c1)
        cov2 = np.matmul(c2.T, c2)
        return cov1, cov2

    def calc_1d_params(self,):
        u_norm = np.abs(self.train())
        c1_u = np.dot(self.c1_data, u_norm)
        c2_u = np.dot(self.c2_data, u_norm)
        c1_u_mu = np.sum(c1_u) / c1_u.shape[0]
        c2_u_mu = np.sum(c2_u) / c2_u.shape[0]

        c1_u_var = np.sum( (c1_u-c1_u_mu)**2 ) / (c1_u.shape[0]-1)
        c2_u_var = np.sum( (c2_u-c2_u_mu)**2 ) / (c2_u.shape[0]-1)
        return c1_u, c2_u, c1_u_mu, c2_u_mu, c1_u_var, c2_u_var

    def get_projected_data(self,):
        u_norm = self.train()

        return


def main():
    # generate samples from 2d gaussians
    mu1 = np.array([0.1, .5])
    mu2 = np.array([2, 2])
    cov1 = np.array([[0.2, 0.05], [0.05, 0.4]])
    cov2 = np.array([[0.4, 0.1], [0.1, 0.1]])
    c1_data, c2_data = generate_data(25, mu1, mu2, cov1, cov2)

    # get sample means/covariances, and train
    fd = fisher_discriminator(c1_data, c2_data)
    sample_mu1, sample_mu2 = fd.calc_means()
    sample_cov1, sample_cov2 = fd.calc_covariances()
    c1_u, c2_u, c1_u_mu, c2_u_mu, c1_u_var, c2_u_var = fd.calc_1d_params()
    u = np.abs(fd.train())

    # plotting ranges for u vector
    x_min = np.min(np.concatenate((c1_data[:,0], c2_data[:,0]), axis=0), axis=0)
    x_max = np.max(np.concatenate((c1_data[:,0], c2_data[:,0]), axis=0), axis=0)
    u_x = np.linspace(x_min, x_max, 10)
    u_y = (u[1]/u[0]) * u_x

    # extract equation for line from beta parameters
    lr = linear_regressor(c1_data, c2_data)
    b = lr.train()
    print('linear regression parameters:\n', b)
    xy_intercept = np.array([b[2], -b[1]])
    xy_intercept_norm = xy_intercept / np.sqrt(np.dot(xy_intercept, xy_intercept))
  
    b_y = -b[1]/b[2] * u_x - b[0]/b[2]
    print(u)
    print(xy_intercept_norm)
    print(np.dot(u.T, xy_intercept_norm))

    # plotting
    plt.plot(u_x, u_y, c='k', label='Fisher')
    plt.plot(u_x, b_y, c='k', linestyle='--', label='Regression')
    plt.scatter(c1_data.T[0], c1_data.T[1], s=60, marker='x', label='class 1')
    plt.scatter(c2_data.T[0], c2_data.T[1], s=60, marker='x', label='class 2')
    plt.scatter(sample_mu1[0], sample_mu1[1], s=50, c='g', marker='o')
    plt.scatter(sample_mu2[0], sample_mu2[1], s=50, c='g', marker='o', label='Sample Mean')
    plt.scatter(mu1[0], mu1[1], s=50, c='r', marker='^')
    plt.scatter(mu2[0], mu2[1], s=50, c='r', marker='^', label='True Mean')
    # projected data
    plt.scatter(c1_u*u[0], c1_u*u[1], s=70, c='b', marker='.')
    plt.scatter(c2_u*u[0], c2_u*u[1], s=70, c='r', marker='.')
    # plt.scatter(c1_u_mu*u[0], c1_u_mu*u[1], s=70, c='c', marker='o', label='Projected Mean')
    plt.scatter((c1_u_mu+c1_u_var)*u[0], (c1_u_mu+c1_u_var)*u[1], s=100, marker='>', c='k')
    plt.xlim((-1, 3.5))
    plt.ylim((-1, 3.5))
    plt.legend()
    plt.grid()
    plt.title('Decision Boundaries')
    plt.show()

    # projected data plotting
    plt.figure()
    plt.scatter(c1_u, np.zeros(c1_u.shape[0]), label='class 1')
    plt.scatter(c1_u_mu, 0, s=200, c='k', marker='^', label='Sample Mean')
    x = np.linspace(-50, 50, 1000)
    plt.plot(x, mlab.normpdf(x, c1_u_mu, np.sqrt(c1_u_var)))
    plt.scatter(c2_u, np.zeros(c2_u.shape[0]), label='class 2')
    plt.scatter(c2_u_mu, 0, s=200, c='k', marker='^')
    x = np.linspace(-10, 10, 1000)
    plt.plot(x, mlab.normpdf(x, c2_u_mu, np.sqrt(c2_u_var)))
    plt.xlim(-1.5, 4.5)
    plt.title('Projected Data and Corresponding Gaussian Distributions')
    plt.xlabel('Distance along u')
    plt.ylabel('probability')
    plt.grid()
    plt.legend()
    plt.show()


    # printing sample stats:
    # print('sample means:')
    # print('class 1 real:', mu1)
    # print('class 1 sample:', sample_mu1)
    # print('difference:', mu1 - sample_mu1)
    # print('sample means:')
    # print('class 2 real:', mu2)
    # print('class 2 sample:', sample_mu2)
    # print('difference:', mu2 - sample_mu2)



if __name__ == "__main__":
    main()
