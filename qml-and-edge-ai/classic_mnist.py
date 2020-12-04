import numpy as np
import scipy
from scipy.linalg import expm
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def c_MNIST(training_size, test_size, n, PLOT_DATA=True):
#    class_labels = [r'Benign', r'Malignant']
    
    # First the dataset must be imported.
    train = pd.read_csv('mnist_test.csv')

    print(train.head())
    # To find if the classifier is accurate, a common strategy is
    # to divide the dataset into a training set and a test set.
    # Here the data is divided into 70% training, 30% testing.
    X_train = train.drop(['label'], axis='columns', inplace=False)
    y_train = train['label']
    X_train, X_test,  Y_train, Y_test = train_test_split(X_train, y_train,  test_size=0.3, random_state=109)
    print(Y_test)
    print('Train: X=%s, y=%s' % (X_train.shape,  Y_train.shape))
    print('Test: X=%s, y=%s' % (X_test.shape, Y_train.shape))
    ##Filter the dataset to keep just the 3s and 6s, remove the other classes.
    #At the same time convert the label, y, to boolean: True for 3 and False for 6. 
    def filter_36(x, y):
        keep = (y == 3) | (y == 6)
        x, y = x[keep], y[keep]
        y = y == 3
        return x,y
####################################################
    # Now the dataset's features will be standardized
    # to fit a normal distribution.
    # Normalize dataset to have 0 unit variance so the pixels from the images 
    #have a very small range and can be     computed efficiently.
    X_train,  Y_train = filter_36(X_train, Y_train)
    X_test, Y_test = filter_36(X_test, Y_test)
    print('After filter: Y_train')
    print(Y_train)
    print('After filter: Y_test')
    print(Y_test)
    print('Train: X=%s, y=%s' % (X_train.shape,  Y_train.shape))
    print('Test: X=%s, y=%s' % (X_test.shape, Y_train.shape))
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # To be able to use this data with the given
    # number of qubits, the data must be broken down from
    # 30 dimensions to `n` dimensions.
    # This is done with Principal Component Analysis (PCA),
    # which finds patterns while keeping variation.
    pca = PCA(n_components=n).fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    print('Train: X=%s, y=%s' % (X_train.shape,  Y_train.shape))
    print('Test: X=%s, y=%s' % (X_test.shape, Y_train.shape))
    # The last step in the data processing is
    # to scale the data to be between -1 and 1
    # Set the range for SVM to -1 and +1 so classification can be done 
    #  based on where a datapoint lies on the range.
    samples = np.append(X_train, X_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    X_train = minmax_scale.transform(X_train)
    X_test = minmax_scale.transform(X_test)

   

    if PLOT_DATA:
        for k in range(0, 2):
            x_axis_data = X_train[Y_train == k, 0][:training_size]
            y_axis_data = X_train[Y_train == k, 1][:training_size]
            
            label = 'Malignant' if k is 1 else 'Benign'
            plt.scatter(x_axis_data, y_axis_data, label=label)

        plt.title("Breast Cancer Dataset (Dimensionality Reduced With PCA)")
        plt.legend()
        plt.show()
        
# train test split
# X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, train_size = 0.2 ,random_state = 10)


    return X_train,  X_test,  Y_train, Y_test
