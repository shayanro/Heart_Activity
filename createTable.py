import sqlite3
import os
import numpy as np  # we all know what np is :)
import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
import csv
import os
import pickle

conn = sqlite3.connect('database.db')
c = conn.cursor()

def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS database(time_stamp REAL, heart_rate REAL, imu1 REAL, imu2 REAL, imu3 REAL)")

create_table()


os.chdir("D:\PAMAP_Dataset\data")
files = os.listdir()
# print(directories)
for file in files:
    print(file)
    with open(file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=' ')
        for row in readCSV:
            time_stamp = row[0]
            if row[2] != 'NaN':
                heart_rate = row[2]
            imu1 = row[4]
            imu2 = row[5]
            imu3 = row[6]

            c.execute("INSERT INTO database(time_stamp ,heart_rate ,imu1 ,imu2 ,imu3) VALUES (?, ?, ?, ?, ?)",
                      (time_stamp, heart_rate, imu1, imu2, imu3))
            conn.commit()

c.close()
conn.close()