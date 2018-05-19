import pandas as pd
import os
import numpy as np


movie_directory  = 'C:\\Users\\MX\\Documents\\Xavier\\CSPrel\\Recommneder\\netflix data\\nf_prize_dataset.tar\\download\\training_set\\'
movie_list = []
#N = 100480507
N = 5010199


movieID = np.zeros((N, 3))
start = 0
i = 0

for file in os.listdir(movie_directory):
    if (file.endswith(".txt")) & (i < 1000):
        mv = pd.read_csv(os.path.join(movie_directory, file), skiprows=[0], names=['userID', 'rating', 'date'])
        end = len(mv) + start
        movieID[start:end, 0] = mv.userID
        movieID[start:end, 1] = mv.rating
        movieID[start:end, 2] = i
        i += 1

        del mv

        start = end

u, indices = np.unique(movieID[:, 0], return_inverse=True)
u = np.arange(u.shape[0])

R = np.zeros((N, 4))
R[:, 0] = u[indices].astype('int32')
R[:, 1:3] = movieID[:, 1:3]
R[:, 2] = R[:, 2].astype('int32')
R[:, 3] = movieID[:, 0]

np.save( 'C:\\Users\\MX\\Documents\\Xavier\\CSPrel\\Recommneder\\netflix data\\short_training.npy', R)






