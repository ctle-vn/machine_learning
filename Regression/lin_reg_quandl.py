#open source, BSD-licensed library providing high-performance,
# easy-to-use data structures and data analysis tools for the Python programming
import pandas as pd
import quandl # marketplace for financial and economic data delivered in modern formats for today's analysts
import math
import numpy as np #arrays
from sklearn import preprocessing  # scaling features between [-1,1]
from sklearn import cross_validation # shuffles data for statistics, unbias sample, seperates data
from sklearn import svm # support vector machine, performs regression
from sklearn.linear_model import LinearRegression
import datetime # datetime objects
import matplotlib.pyplot as plt
from matplotlib import style
import pickle #module that allows user to save a classifier to save time constantly training classifiers.

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
#           Price           x           x           x
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]  # Features, attributes of what in our mind what may cause the Adjusted Close price in 10 days to change

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) #replace nand data with some value

forecast_out = int(math.ceil(0.1*len(df)))

#Labels.
df['label'] = df[forecast_col].shift(-forecast_out) #shifting columns negatively ( up )

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X) # scaling X before we feed it to classifier
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

#used to fit our classifier
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2) #takes all features and labels, shuffle, outputs training/testing data
'''
#this block of code is ran once to train the classifier, then saved into a pickle object.
#define and train classifier
classifier = LinearRegression(n_jobs=-1)
classifier.fit(X_train, y_train) #train classifier on data
#uses pickle
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(classifier, f) #dumps trained classifier into temp var 'f'
'''

pickle_in = open('linearregression.pickle', 'rb') #use classifier by opening pickle file
classifier = pickle.load(pickle_in) #loads pickle object

accuracy = classifier.score(X_test, y_test) #test on diff data because classifier would know from previously trained data

forecast_set = classifier.predict(X_lately)

#forecast_set is an array of forecasts, showing that not only could you just
#seek out a single prediction, but you can seek out many at once.
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


# iterating through the forecast set, taking each forecast and day, and then
# setting those values in the dataframe (making the future "features" NaNs).
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
