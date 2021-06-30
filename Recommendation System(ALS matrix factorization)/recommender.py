import sys
import time
import numpy as np
import pandas as pd


class ALS_MatrixFactorizer():
    def __init__(self, R, nf, reg_param, epochs):
        """
        R : rating matrix
        nf : dim of latent matrix
        _lambda : for normalization
        epochs : max iteration
        X user, Y item
        """
        self.R = R
        self.nf = nf
        self._lambda = reg_param
        self.epochs = epochs

    def fit(self):
        self.X = np.random.normal(size=(self.R.shape[0], self.nf))
        self.Y = np.random.normal(size=(self.R.shape[1], self.nf))

        for epoch in range(self.epochs):
            start_epoch = time.time()
            self.optimize_X()
            self.optimize_Y()

            RMSE = self.train_RMSE()
            print("Epoch %d / train_RMSE = %.4f" % (epoch + 1, RMSE))

            print(str(epoch + 1) + " is     done!")
            print("time spent :      %d s" % (time.time()-start_epoch))

    def optimize_X(self):
        for i, Ri in enumerate(self.R):
            front = np.linalg.inv(
                np.dot(self.Y.T, np.dot(np.diag(Ri), self.Y)) + self._lambda * np.identity(self.nf))
            rear = np.dot(self.Y.T, np.dot(np.diag(Ri), self.R[i].T))
            self.X[i] = np.matmul(front, rear)

    def optimize_Y(self):
        for j, Rj in enumerate(self.R.T):
            front = np.linalg.inv(
                np.dot(self.X.T, np.dot(np.diag(Rj), self.X)) + self._lambda * np.identity(self.nf))
            rear = np.dot(self.X.T, np.dot(np.diag(Rj), self.R[:, j]))
            self.Y[j] = np.matmul(front, rear)

    def train_RMSE(self):
        Ri, Rj = self.R.nonzero()
        cost = 0
        for i, j in zip(Ri, Rj):
            cost += pow(self.R[i, j] - self.X[i, :].dot(self.Y[j, :].T), 2)
        return np.sqrt(cost/len(Ri))


if __name__ == '__main__':
    start = time.time()
    # check correct parameter
    if len(sys.argv) != 3:
        print("Insufficient arguments")
        sys.exit()

    training_data = sys.argv[1]
    test_data = sys.argv[2]

    train_df = pd.read_table(training_data, sep="\t", names=[
        'user_id', 'item_id', 'rating', 'time_stamp'])
    train_df.drop('time_stamp', axis=1, inplace=True)

    rating_matrix = train_df.pivot_table(
        'rating', index='user_id', columns='item_id').fillna(0).to_numpy()

    matrix_factorizer = ALS_MatrixFactorizer(
        R=rating_matrix, nf=3, reg_param=0.1, epochs=20)

    print(" 학습 시작 ")
    matrix_factorizer.fit()
    res = matrix_factorizer.X.dot(matrix_factorizer.Y.T)

    output_file = open("u" + str(sys.argv[1][1]) + ".base_prediction.txt", "w")
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            print(str(i+1)+'\t'+str(j+1)+'\t'+str(res[i][j]), file=output_file)

    print("time:  ", str(time.time()-start))
