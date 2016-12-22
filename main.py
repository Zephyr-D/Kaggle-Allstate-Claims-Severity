
# -------------------------------------------------
# CSC 576 Final Project (group)
# Li Ding
# Nov. 2016
# Problem: Allstate Claims Severity (Kaggle Competition)
# -------------------------------------------------

import pandas
import numpy

# numpy.set_printoptions(threshold=np.inf)


# ----------- Data input --------------
# Read raw data from the file
dataset = pandas.read_csv("train.csv")
dataset_test = pandas.read_csv("test.csv")
dataset = dataset.iloc[: ,1:]
dataset_test = dataset_test.iloc[: ,1:]

# Print all rows and columns. Dont hide any
# pandas.set_option('display.max_rows', None)
# pandas.set_option('display.max_columns', None)


# ----------- Data preparation --------------
# cat1 to cat116 have strings. The ML algorithms we are going to study require numberical data
# One-hot encoding converts an attribute to a binary vector

# Variable to hold the list of variables for an attribute in the train and test data
labels = []

# range of features considered
split = 116

# names of all the columns
cols = dataset.columns

for i in range(0 ,split):
    train = dataset[cols[i]].unique()
    test = dataset_test[cols[i]].unique()
    labels.append(list(set(train) | set(test)))

# create a dataframe with only continuous features data= dataset.iloc[:, split:]


# log1p function applies log(1+x) to all elements of the column
# dataset["loss"] = numpy.log(dataset["loss"] + 10)


def exp(np):
    return numpy.exp(np) - 10


# Import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# One hot encode all categorical attributes
cats = []
for i in range(0, split):
    # Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset.iloc[:, i])
    feature = feature.reshape(dataset.shape[0], 1)
    # One hot encode
    onehot_encoder = OneHotEncoder(sparse=False, n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

# Make a 2D array from a list of 1D arrays
encoded_cats = numpy.column_stack(cats)

# Print the shape of the encoded data
print(encoded_cats.shape)

# Concatenate encoded attributes with continuous attributes
dataset_encoded = numpy.concatenate((encoded_cats, dataset.iloc[:, split:].values), axis=1)
del cats
del feature
del dataset
del encoded_cats
print(dataset_encoded.shape)

# get the number of rows and columns
r, c = dataset_encoded.shape

# create an array which has indexes of columns
i_cols = []
for i in range(0, c - 1):
    i_cols.append(i)

# Y is the target column, X has the rest
X = dataset_encoded[:, 0:(c - 1)]
Y = dataset_encoded[:, (c - 1)]
del dataset_encoded


def MAE(Y, YP):
    s = 0
    for i in range(len(Y)):
        s += abs(Y[i] - YP[i])
    return s / len(Y)


# ------------ Linear Regression ------------
'''
from sklearn.linear_model import LinearRegression

# Set the base model
model = LinearRegression(n_jobs=-1)
algo = "LR"

# Accuracy of the model using all features
for name,i_cols_list in X_all:
    model.fit(X_train[:,i_cols_list],Y_train)
    result = MAE(exp(Y_val), exp(model.predict(X_val[:,i_cols_list])))
    mae.append(result)
    print(name + " %s" % result)
comb.append(algo)

# Result obtained after running the algo. Comment the below two lines if you want to run the algo
#mae.append(1278)
#comb.append("LR" )

print mae
'''

# ------------ l1 loss -------------
'''
from sklearn.linear_model import SGDRegressor

l1 = SGDRegressor(loss = 'epsilon_insensitive', epsilon = 0, penalty = 'l1', n_iter = 1, learning_rate = 'constant', eta0 = 0.0005)

# Initialization
coef = numpy.loadtxt('l1_coef')
inter = numpy.loadtxt('l1_inter')
ny = len(Y)
mae = 2000
#l1.fit(X, Y)
#print sum(abs(Y - l1.predict(X)))/len(Y)
#coef = l1.coef_
#inter = l1.intercept_

# SGD iteration
for i in range(50):
	l1.fit(X, Y, coef_init = coef, intercept_init = inter)
	p = l1.predict(X)
	for i in p:
		if i < 0:
			i = 5
	if mae > sum(abs(Y - p))/ny:
		mae = sum(abs(Y - p))/ny
		coef = l1.coef_
		inter = l1.intercept_
	print mae

# Save result
numpy.savetxt('l1_coef', coef)
numpy.savetxt('l1_inter', inter)
numpy.savetxt('l1_mae', numpy.array([mae]))

prediction = l1.predict(X_test)
'''

# ------------------ l1 loss with some interaction -------------------
'''
nc = len(X[0,:])
for i in range(1176, nc):
	for j in range(i,nc):
		X = numpy.hstack((X, numpy.array(X[:,i] * X[:,j])[...,None]))

from sklearn.linear_model import SGDRegressor

l1 = SGDRegressor(loss = 'epsilon_insensitive', epsilon = 0, penalty = 'l1', n_iter = 10, learning_rate = 'constant', eta0 = 0.001, alpha = 0.0001)

# Initialization
coef = numpy.loadtxt('l1x_coef')
inter = numpy.loadtxt('l1x_inter')
ny = len(Y)
mae = 2000
#l1.fit(X, Y)
#print sum(abs(Y - l1.predict(X)))/len(Y)
#coef = l1.coef_
#inter = l1.intercept_

# SGD iteration
for i in range(10):
	l1.fit(X, Y, coef_init = coef, intercept_init = inter)
	p = l1.predict(X)
	for i in p:
		if i < 0:
			i = 5
	if mae > sum(abs(Y - p))/ny:
		mae = sum(abs(Y - p))/ny
		coef = l1.coef_
		inter = l1.intercept_
	print mae

# Save result
numpy.savetxt('l1x_coef', coef)
numpy.savetxt('l1x_inter', inter)
numpy.savetxt('l1x_mae', numpy.array([mae]))

'''

# ---------------- squareroot y ------------------

from sklearn.linear_model import SGDRegressor

ls = SGDRegressor(loss='squared_loss', penalty='l1', n_iter=1, learning_rate='constant', eta0=0.00001, alpha=0.000001)

# Initialization
coef = numpy.loadtxt('ls_coef')
inter = numpy.loadtxt('ls_inter')
ny = len(Y)
mae = 2000
ls.fit(X, numpy.sqrt(Y))
print sum(abs(Y - numpy.square(ls.predict(X)))) / len(Y)
coef = ls.coef_
inter = ls.intercept_

# SGD iteration
for i in range(100):
    ls.fit(X, numpy.sqrt(Y), coef_init=coef, intercept_init=inter)
    err = sum(abs(Y - numpy.square(ls.predict(X)))) / ny
    print err
    if mae > err:
        mae = err
        coef = ls.coef_
        inter = ls.intercept_
        lopt = ls
    print mae

# Save result
numpy.savetxt('ls_coef', coef)
numpy.savetxt('ls_inter', inter)
numpy.savetxt('ls_mae', numpy.array([mae]))

prediction = numpy.square(lopt.predict(X_test))

# ---- xgb --------

import xgboost as xgb

from sklearn.metrics import mean_absolute_error

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test['loss'] = np.nan
joined = pd.concat([train, test])


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'mae', mean_absolute_error(np.exp(preds), np.exp(labels))


if __name__ == '__main__':
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


            joined[column] = joined[column].apply(lambda x: filter_cat(x), 1)

        joined[column] = pd.factorize(joined[column].values, sort=True)[0]

    train = joined[joined['loss'].notnull()]
    test = joined[joined['loss'].isnull()]

    shift = 200
    y = np.log(train['loss'] + shift)
    ids = test['id']
    X = train.drop(['loss', 'id'], 1)
    X_test = test.drop(['loss', 'id'], 1)

    RANDOM_STATE = 2016
    params = {
        'min_child_weight': 1,
        'eta': 0.01,
        'colsample_bytree': 0.5,
        'max_depth': 12,
        'subsample': 0.8,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'verbose_eval': True,
        'seed': RANDOM_STATE
    }

    xgtrain = xgb.DMatrix(X, label=y)
    xgtest = xgb.DMatrix(X_test)

    model = xgb.train(params, xgtrain, int(2012 / 0.9), feval=evalerror)

    prediction = np.exp(model.predict(xgtest)) - shift

# ----- Testing and Submission -------------------


dataset_test = pandas.read_csv("test.csv")
# Drop unnecessary columns
ID = dataset_test['id']
dataset_test.drop('id', axis=1, inplace=True)

# One hot encode all categorical attributes
cats = []
for i in range(0, split):
    # Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset_test.iloc[:, i])
    feature = feature.reshape(dataset_test.shape[0], 1)
    # One hot encode
    onehot_encoder = OneHotEncoder(sparse=False, n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

# Make a 2D array from a list of 1D arrays
encoded_cats = numpy.column_stack(cats)

del cats

# Concatenate encoded attributes with continuous attributes
X_test = numpy.concatenate((encoded_cats, dataset_test.iloc[:, split:].values), axis=1)
del encoded_cats
del dataset_test

# Write submissions to output file in the correct format
with open("submission.csv", "w") as subfile:
    subfile.write("id,loss\n")
    for i, pred in enumerate(list(prediction)):
        if pred <= 5:
            subfile.write("%s,%s\n" % (ID[i], 5))
        elif pred >= 100000:
            subfile.write("%s,%s\n" % (ID[i], 100000))
        else:
            subfile.write("%s,%s\n" % (ID[i], pred))

















