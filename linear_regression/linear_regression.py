import numpy as np
import matplotlib.pyplot as plt 

class linear_regressor():
    def __init__(self, fname_train, fname_test):
        self.load_data(fname_train, fname_test)
        self.B = None

    def load_data(self, fname_train, fname_test=None):
        train = np.genfromtxt(fname_train, delimiter='\t')
        self.x_train = train[:, 1:9]
        self.y_train = train[:, 9]

        # change this if not treating all data as training data
        test = np.genfromtxt(fname_test, delimiter='\t')
        self.x_test = None
        self.y_test = None
        return

    def train(self, columns, reg=0.0):
        # use specified columns for model fitting
        X = np.zeros(shape=(self.x_train.shape[0], len(columns)+1))
        for i in range(0, len(columns)):
            X[:,i+1] = self.x_train[:, int(columns[i])]

        # normalize data to have 0 mean and unit variance
        # X = self.normalize(X)
        # y = self.normalize(self.y_train)
        y = self.y_train

        # insert column of 1s at beginning of data to allow
        X[:,0] = 1.0
        self.X = X

        # B = (Xt X)^-1 Xt y
        Xt_X_inv = np.linalg.inv(np.matmul(X.T, X) + np.eye(len(columns)+1)*reg)
        self.B = np.dot(np.matmul(Xt_X_inv, X.T), y)
        return self.B

    def normalize(self, X):
        """ normalize a dataset to have 0-mean and unit variance """
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    def training_loss(self,):
        assert self.B is not None, "must train classifier prior to determining loss"
        predictions = np.dot(self.X, self.B)
        loss = np.sum(np.square(predictions - self.y_train)) / predictions.shape[0]
        return loss, predictions

def main():
    reg = linear_regressor('data.txt', 'data.txt')
    B = reg.train(columns=np.linspace(0, 7, 8))
    loss, predictions = reg.training_loss()
    predictors = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    print('\nloss for using all predictors:\n', loss)
    print('\nB values:\n', B)

    losses = []
    # each column individually
    for i in range(0, 8):
        B = reg.train(columns=np.array([i]))
        loss, predictions = reg.training_loss()
        losses.append(loss)
    print('\nindividual feature losses:\n', losses)
    plt.figure()
    plt.bar(x=np.linspace(0, len(losses), len(losses)), height=losses)
    plt.xticks(np.linspace(0, len(losses), len(losses)), predictors)
    plt.title('Training MSE for Single Predictor Regression Models')
    plt.show()

    # each column sequentially
    losses = []
    for i in range(0, 8):
        if i == 0:
            B = reg.train(columns=[0])        
        else:
            B = reg.train(columns=np.linspace(0,i,i+1))
        loss, predictions = reg.training_loss()
        losses.append(loss)
    print('\nsequentially adding features losses:\n', losses)
    plt.figure()
    plt.plot(losses)
    plt.xticks(np.linspace(0, len(losses), len(losses)+1), predictors)
    plt.grid()
    plt.title('Training MSE for Sequentially Adding Predictors')
    plt.show()

if __name__ == "__main__":
    main()














