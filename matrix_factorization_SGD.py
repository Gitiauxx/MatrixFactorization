import numpy as np
import pandas as pd
from numba import jit


class matrix_factorization(object):

    def __init__(self, R, n_users, n_items, k, eta, beta):

        self.R = R
        self.eta = eta
        self.beta = beta
        self.n = len(self.R)
        self.U = np.random.normal(scale=1 / k, size=(n_users, k))
        self.V = np.random.normal(scale=1 / k, size=(n_items, k))

    def split_train_test(self):
        _, count_elements = np.unique(self.R[:, 0], return_counts=True)
        _, indices = np.unique(self.R[:, 0], return_inverse=True)
        number_duplicates = count_elements[indices]

        test_list = []

        for i in range(self.n):
            if (i%10 == 0) & (number_duplicates[i] > 1):
                test_list.append(i)
        test_array = np.array(test_list)
        mask = np.ones(self.n, dtype=bool)
        mask[test_array] = False

        return R[test_array], R[mask]

    def gradient_descent(self, R):

        for i in range(R.shape[0]):
            user = int(R[i, 0])
            item = int(R[i, 2])
            error = R[i, 1] - np.dot(self.U[user, :], self.V[item, :].transpose())
            update_u = self.V[item, :] * error - \
                              self.beta * self.U[user, :]
            update_v = self.U[user, :] * error - \
                              self.beta * self.V[item, :]
            self.U[user, :] += self.eta * update_u
            self.V[item, :] += self.eta * update_v

            self.squared_error += error**2

    def iteration(self, nepoch=10, method='MF'):

        test, train = self.split_train_test()
        print("The size of the train sample is {}".format(train.shape[0]))
        print("The size of the test sample is {} \n".format(test.shape[0]))

        for epoch in range(nepoch):
            self.squared_error = 0
            np.random.shuffle(self.R)

            self.gradient_descent(train)
            rmse_test = self.mse_test(test)

            rmse_train = self.mse(train.shape[0])
            print("At iteration {} the rmse is equal to {:4f}".format(epoch, rmse_train))
            print("At iteration {} the rmse in the test sample is equal to {:4f} \n".format(epoch, rmse_test))

        report = pd.DataFrame(index=np.arange(nepoch))
        report['rmse_train'] = rmse_train
        report['rmse_test'] = rmse_test
        report['method'] = method + '_' + str(self.U.shape[1])

        report.to_csv('C:\\Users\\MX\\Documents\\Xavier\\CSPrel\\Recommneder\\NetFlix_Test\\report\\rmse_' + method + str(self.U.shape[1]) + '.csv')


    def mse(self, size):
        return np.sqrt(self.squared_error / size)

    def mse_test(self, test):
        error = 0
        size = test.shape[0]
        for i in range(size):
            user = int(test[i, 0])
            item = int(test[i, 2])
            error += (test[i, 1] - self.get_rating_item_user(user, item)) ** 2

        return np.sqrt(error / size)

    def get_rating_item_user(self, user, item):
        return np.dot(self.U[user, :], self.V[item, :].transpose())

    def get_rating(self):
        return np.dot(self.U, self.V.T)

class matrix_factorization_bias(matrix_factorization):

    def __init__(self, R, n_users, n_items, k, eta, beta):
        super().__init__(R, n_users, n_items, k, eta, beta)
        self.A = np.random.normal(scale=1, size=n_users)
        self.B = np.random.normal(scale=1, size=n_items)

    def gradient_descent(self, R):

        for i in range(R.shape[0]):
            user = int(R[i, 0])
            item = int(R[i, 2])
            error = R[i, 1] - np.dot(self.U[user, :], self.V[item, :].transpose()) - \
                    self.A[user] - self.B[item]

            update_u = self.V[item, :] * error - \
                              self.beta * self.U[user, :]
            update_v = self.U[user, :] * error - \
                              self.beta * self.V[item, :]

            update_a = error - self.beta * self.A[user]
            update_b = error - self.beta * self.B[item]

            self.U[user, :] += self.eta * update_u
            self.V[item, :] += self.eta * update_v
            self.A[user] += self.eta * update_a
            self.B[item] += self.eta * update_b

            self.squared_error += error**2

    def get_rating_item_user(self, user, item):
        return np.dot(self.U[user, :], self.V[item, :].transpose()) + self.A[user] + self.B[item]

class matrix_factorization_plus(matrix_factorization_bias):

    def __init__(self, R, n_users, n_items, k, eta, beta):
        super().__init__(R, n_users, n_items, k, eta, beta)
        self.Y = np.random.normal(scale=1 / k, size=(n_items, k))
        self.user_dict = self.get_user_dict(self.R)

    def get_user_dict(self, R):
        user_dict = {}

        for i in range(R.shape[0]):
            user = R[i, 0]
            item = R[i, 2]

            if user not in user_dict:
                user_dict[user] = [int(item)]
            else:
                user_dict[user].append(int(item))

        return user_dict

    def gradient_descent(self, R):

        user_dict = self.get_user_dict(R)

        for i in range(R.shape[0]):
            user = int(R[i, 0])
            item = int(R[i, 2])
            number_items = len(user_dict[user])

            watched = user_dict[user]
            y_sum = 0
            for item_watched in watched:
                y_sum += self.Y[item_watched, :]

            error = R[i, 1] - np.dot(self.U[user, :] + 1 / number_items * y_sum, self.V[item, :].transpose()) - \
                    self.A[user] - self.B[item]

            update_u = self.V[item, :] * error - \
                              self.beta * self.U[user, :]
            update_v = (self.U[user, :] + 1 / number_items * y_sum) * error - \
                              self.beta * self.V[item, :]
            update_y = 1 / number_items * self.V[item, :] * error - self.beta * self.Y[item, :]

            update_a = error - self.beta * self.A[user]
            update_b = error - self.beta * self.B[item]

            self.U[user, :] += self.eta * update_u
            self.V[item, :] += self.eta * update_v
            self.A[user] += self.eta * update_a
            self.B[item] += self.eta * update_b
            self.Y[item, :] += self.eta * update_y

            self.squared_error += error**2

    def get_rating_item_user(self, user, item):
        y_factor = 0
        number = 0
        for j in self.user_dict[user]:
            y_factor += self.Y[j, :]
            number += 1

        return np.dot(self.U[user, :] + y_factor / number, self.V[item, :].transpose()) + self.A[user] + self.B[item]


class matrix_factorization_numba(matrix_factorization_bias):
    def gradient_descent(self, R):
        self.squared_error, self.U, self.V, self.A, self.B =_gradient_descent(R, self.U, self.V, self.A,
                                                                              self.B, self.eta, self.beta)

class matrix_factorization_numba_plus(matrix_factorization_plus):

    def __init__(self, R, n_users, n_items, k, eta, beta):
        super().__init__(R, n_users, n_items, k, eta, beta)
        self.Y = np.random.normal(scale=1 / k, size=(n_users, k))

    def update_y(self, R):
        user_dict = self.get_user_dict(R)

        for user in user_dict:
            user = int(user)
            watched = user_dict[user]
            number_items = len(user_dict[user])

            y_sum = 0
            for item_watched in watched:
                y_sum += self.V[item_watched, :]

            self.Y[user, :] = self.Y[user, :] + self.eta * y_sum / number_items


    def gradient_descent(self, R):
        self.squared_error, self.U, self.V, self.A, self.B = _gradient_descent_plus(R, self.U, self.V, self.A,
                                                                               self.B, self.Y, self.eta, self.beta)
        self.update_y(R)

    def get_rating_item_user(self, user, item):
        return np.dot(self.U[user, :] + self.Y[user, :], self.V[item, :].transpose()) + self.A[user] + self.B[item]

@jit(nopython=True)
def _gradient_descent(R, U, V, A, B, eta, beta):
    squared_error = 0
    for i in range(R.shape[0]):
        user = int(R[i, 0])
        item = int(R[i, 2])
        error = R[i, 1] - np.dot(U[user, :], V[item, :].transpose()) - \
                A[user] - B[item]

        update_u = V[item, :] * error - beta * U[user, :]
        update_v = U[user, :] * error - beta * V[item, :]
        update_a = error - beta * A[user]
        update_b = error - beta * B[item]

        U[user, :] += eta * update_u
        V[item, :] += eta * update_v
        A[user] += eta * update_a
        B[item] += eta * update_b
        squared_error += error ** 2

    return squared_error, U, V, A, B

@jit(nopython=True)
def _gradient_descent_plus(R, U, V, A, B, Y, eta, beta):
    squared_error = 0
    for i in range(R.shape[0]):
        user = int(R[i, 0])
        item = int(R[i, 2])
        error = R[i, 1] - np.dot(U[user, :] + Y[user, :], V[item, :].transpose()) - A[user] - B[item]

        update_u = V[item, :] * error - beta * U[user, :]
        update_v = U[user, :] * error - beta * V[item, :]
        update_a = error - beta * A[user]
        update_b = error - beta * B[item]

        U[user, :] += eta * update_u
        V[item, :] += eta * update_v
        A[user] += eta * update_a
        B[item] += eta * update_b
        squared_error += error ** 2

    return squared_error, U, V, A, B



if __name__ == '__main__':
    R = np.load('C:\\Users\\MX\\Documents\\Xavier\\CSPrel\\Recommneder\\netflix data\\short_training.npy')

    n_users = np.unique(R[:, 0]).shape[0]
    n_items = np.unique(R[:, 2]).shape[0]

    print("The number of items is {} and the number of users is {} \n".format(n_items, n_users))

    mf = matrix_factorization_numba_plus(R, n_users, n_items, 10, 0.05, 0.2)
    mf.iteration(10, method='MF_numba_bias')




