import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def smoothing(y,w): # w even!
    target = np.zeros((1,len(y)))
    target[0,0:int(w/2)] = y[0:int(w/2)]
    for i in range(int(w/2),len(y)-int(w/2)):
        target[0,i] = np.mean(y[i-int(w/2):i+int(w/2)])
    return target

# data import
data = pd.read_csv("df_xdata.csv",header = 0, delimiter = ';')
data.drop(["Timestamp"], axis=1, inplace=True)
rezult = pd.read_csv('df_ts.csv', header = 0, delimiter = ';')
sensors = list(data.columns.values)
x = np.arange(len(rezult['ts']))
y = np.array(rezult['ts'])

# removing noise from rezult
w = 4
target = smoothing(y,w)

# removing sensors with std = 0
mean_sensors = np.array([np.mean(data[sensor]) for sensor in sensors])
std_sensors = np.array([np.std(data[sensor]) for sensor in sensors])
std_notzero = np.argwhere(std_sensors)
std_zero = np.argwhere(std_sensors == 0)
for i in range(len(std_zero)):
    data.drop([sensors[std_zero[i,0]]], axis=1, inplace=True)

# removing sensors with large covariance coeff
sensors = list(data.columns.values)
mean_sensors = np.array([np.mean(data[sensors[i]]) for i in range(len(sensors))])
std_sensors = np.array([np.std(data[sensors[i]]) for i in range(len(sensors))])
std_large = np.argwhere((std_sensors/mean_sensors > 1)|(std_sensors > 10000))
for i in range(len(std_large)):
    data.drop([sensors[std_large[i,0]]], axis=1, inplace=True)

# sensors data smoothing
snr = []
for sensor in list(data.columns.values):
    pl,pu = np.percentile(data[sensor],[5,95])
    for j in range(len(data[sensor])):
        if (data[sensor][j] < pl)|(data[sensor][j] > pu):
            if j != 0:
                data[sensor][j] = data[sensor][j-1]
            else:
                data[sensor][j] = np.mean(data[sensor])
    if (pl != 0) and(10*np.log10(pu/pl) < 5):
        snr.append(10*np.log10(pu/pl))
        data[sensor] = pd.Series(smoothing(np.array(data[sensor]),w)[0])
    else: snr.append(10*np.log10(pu))

# data scaling    
sc = StandardScaler()
data_sc = sc.fit_transform(data)
x = np.arange(len(rezult['ts']))
y = np.array(rezult['ts'])

# model constructing, training and estimating
X_train, X_test, y_train, y_test = train_test_split(data_sc, target[0], test_size=0.85, random_state=0)
reg1 = GradientBoostingRegressor(random_state=1, n_estimators=150)
reg2 = RandomForestRegressor(random_state=241,max_depth = 10, n_estimators=100)
reg3 = LinearRegression()
ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
kf = KFold(n_splits=6,shuffle=True,random_state=1)
scores = cross_val_score(ereg, data_sc, target[0], cv=kf,scoring = 'r2')
print("Оценки на кросс-валидации: ", scores)
print("Средняя оценка на кросс-валидации: {:.2f}".format(np.mean(scores)))


fig, ax = plt.subplots(2)
ax[0].plot(x,rezult['ts'],x,target[0])
ax[0].set_title('Целевая переменная и её сглаживание')
x = np.arange(len(target[0]))
eregf = ereg.fit(X_train, y_train)
y_pred = eregf.predict(data_sc)
reg1.fit(X_train, y_train)
print(reg1.feature_importances_)
print(eregf.score(data_sc,target[0]))
ax[1].plot(x,y_pred,x,target[0])
ax[1].set_title('Результат')
ax[1].legend(['Прогноз','Реальное значение'])
plt.show()

