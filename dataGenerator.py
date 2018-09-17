import os
import numpy as np  # we all know what numpy is :)
from sklearn.preprocessing import normalize
import csv
import os
import pickle

os.chdir("D:\PAMAP_Dataset")

index = 0
file = 'database.csv'
with open(file) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    heart_rates = []
    imus = []
    for row in readCSV:
        # print(row)
        # time_stamp = row[0]
        heart_rate = row[1]
        imu = [row[2], row[3], row[4]]

        # time_stamps.append(time_stamp)
        heart_rates.append(heart_rate)
        imus.append(imu)
        index += 1

# time_stamp = np.asarray(time_stamps)
heart_rate = np.asarray(heart_rates)
imu = np.asarray(imus)
print('index is: ', index)
# time_stamp = np.delete(time_stamp, np.s_[::2])
heart_rate = np.delete(heart_rate, np.s_[::2])
imu = np.delete(imu, np.s_[::2], axis=0)
imu = normalize(imu, axis=1)
imu = np.reshape(imu, -1)
index = index // 2

T = int(0.1 * 60) # 5 minutes
length = index - 3 * T * 50
print('length is: ', length)
data = np.zeros((length, 3*T*50 + 2))
X = np.zeros((length, 3*T*50 + 1))
y = np.zeros((length, 1))
print(data[1, 2:].shape)
print(imu[0: 1*3*T*50].shape)
for i in range(length):
    print(i)
    data[i, 0] = heart_rate[i+T]
    data[i, 1] = heart_rate[i]
    data[i, 2:] = imu[i:3*T*50+i]

np.random.shuffle(data)

y[:, 0] = data[:, 0]
X[:, :] = data[:, 1:]

with open('X.pickle', 'wb') as f:
    pickle.dump(X, f)
with open('y.pickle', 'wb') as f:
    pickle.dump(y, f)
