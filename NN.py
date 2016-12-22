import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler
import itertools



shift = 0
COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,' \
               'cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,' \
               'cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,' \
               'cat4,cat14,cat38,cat24,cat82,cat25'.split(',')

def encode(charcode):
    r = 0
    ln = len(str(charcode))
    for i in range(ln):
        r += (ord(str(charcode)[i]) - ord('A') + 1) * 26 ** (ln - i - 1)
    return r


def mungeskewed(train, test, numeric_feats):
    ntrain = train.shape[0]
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, ntrain


print('\nStarted')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

numeric_feats = [x for x in train.columns[1:-1] if 'cont' in x]
categorical_feats = [x for x in train.columns[1:-1] if 'cat' in x]
train_test, ntrain = mungeskewed(train, test, numeric_feats)

# taken from Vladimir's script (https://www.kaggle.com/iglovikov/allstate-claims-severity/xgb-1114)
for column in list(train.select_dtypes(include=['object']).columns):
    if train[column].nunique() != test[column].nunique():
        set_train = set(train[column].unique())
        set_test = set(test[column].unique())
        remove_train = set_train - set_test
        remove_test = set_test - set_train

        remove = remove_train.union(remove_test)


        def filter_cat(x):
            if x in remove:
                return np.nan
            return x


        train_test[column] = train_test[column].apply(lambda x: filter_cat(x), 1)


print('')
for comb in itertools.combinations(COMB_FEATURE, 2):
    feat = comb[0] + "_" + comb[1]
    train_test[feat] = train_test[comb[0]] + train_test[comb[1]]
    train_test[feat] = train_test[feat].apply(encode)
    print('Combining Columns:', feat)


print('')
for col in categorical_feats:
    print('Analyzing Column:', col)
    train_test[col] = train_test[col].apply(encode)

print(train_test[categorical_feats])

ss = StandardScaler()
train_test[numeric_feats] = \
    ss.fit_transform(train_test[numeric_feats].values)

train = train_test.iloc[:ntrain, :].copy()
test = train_test.iloc[ntrain:, :].copy()

print('\nMedian Loss:', train.loss.median())
print('Mean Loss:', train.loss.mean())

train_y = np.array(train['loss'])
train_x = np.array(train.drop(['loss','id'], axis=1))


np.savetxt('train_x', train_x)
np.savetxt('train_y', train_y)




train_y = np.loadtxt('train_y')
train_x = np.loadtxt('train_x')


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l1l2,l1,l2
from keras.models import load_model


model = Sequential()
model.add(Dense(1024, input_dim=725, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1, activation='linear'))

p = 4000

sgd = SGD(lr=1e-8, momentum=0.7, nesterov=True)
model.compile(loss='mae', optimizer=sgd)

for i in range(50):
    model.load_weights('nn_2048')
    k = 1
    h = model.fit(train_x, train_y, nb_epoch=k, batch_size=1000, validation_split=0.1, shuffle=True,verbose=2)
    if h.history['val_loss'][k-1] < p:
        model.save('nn_2048')
        p = h.history['val_loss'][k-1]
        print str(i) + ': Weights updated. Loss: ', str(h.history['loss'][k-1]), str(p)
    else:
        print str(i) + ': Not improving.'



#score = model.evaluate(train_x, train_y, batch_size=10000)
#model.save("3l-" + str(int(score)))
a = model.predict(train_x, batch_size=10000)
a = np.reshape(a,(188318))
np.mean(abs(np.exp(a) - np.exp(train_y)))





