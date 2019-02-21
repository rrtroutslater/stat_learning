import numpy as np 
import matplotlib.pyplot as plt 


class fisher_discriminator():
    def __init__(self, fname_train):
        self.training_data = self.load_data(fname_train)

        # self.class_covariances = None
        # self.class_means = self.calc_means()
        # self.class_determinants = None


    def train(self,):
        covs = self.calc_covariances()

        return


    def calc_means(self,):
        means = {}

        for key in self.training_data.keys():
            mu = np.sum(self.training_data[key], axis=0)    # sum over data
            mu = mu / self.training_data[key].shape[0]      # divide by N
            means[key] = mu

        print(means)
        return means


    def calc_covariances(self,):
        covs = {}
        means = self.calc_means()

        for key in means.keys():
            covs[key] = np.matmul(self.training_data[key].T, self.training_data[key])
            print(covs[key].shape)

        return covs


    def calc_determinants(self,):
        dets = {}
        covs = self.calc_covariances()

        for key in covs.keys():
            dets[key] = np.linalg.det(covs[key])
            print(dets[key])

        return


    def load_data(self, fname):
        data_raw = np.genfromtxt(fname, delimiter=' ')
        data = {i : [] for i in range(0, 10)}

        for row in data_raw:
            data[int(row[0])].append(row[1:])

        for key in data.keys():
            data[key] = np.array(data[key])

        return data


def main():
    fd = fisher_discriminator('zip.train')
    fd.calc_determinants()
    means = fd.calc_means()
    print(means)


if __name__ == "__main__":
    main()
