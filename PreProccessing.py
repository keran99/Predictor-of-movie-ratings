from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
import pandas as pd

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import decomposition

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def preprocessing(df, dozScore, varianceThreshold, percentage, doPca, components, dol2Norm):

    dataset = manageMissingValues(df)
    X = dataset.drop("AVG_RATING", 1)
    Y = dataset.AVG_RATING

    # Normalization
    X = scalingDataset(X, dozScore)
    # We need to do feature selection
    X, Y = featureSelection(X, Y, varianceThreshold, percentage)
    if (dol2Norm == True):
        X =pd.DataFrame(L2_reg(np.array(X)))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=7)

    # Dimensionality reduction
    if (doPca == True):
        X_train, X_test = principalComponentAnalysis(X_train, X_test, components)

    X_final = np.concatenate((X_train, X_test))
    y_final = np.concatenate((y_train,y_test))


    return X_final,y_final,X_train, X_test, y_train, y_test


def scalingDataset(X, zscore):
    if (zscore == True):
        # Z normalization
        zscaler = StandardScaler()
        Xz = zscaler.fit_transform(X)
        X = pd.DataFrame(Xz, columns=X.columns)
    else:
        # Normalization
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_ = scaler.fit_transform(X)
        X = pd.DataFrame(X_, columns=X.columns)
    return X


def manageMissingValues(dataset):
    cleaned_dataset = dataset.dropna()
    return cleaned_dataset


def featureSelection(X, Y, varianceTreshold, percentage):
    print('*************************** Feature Selection -  BEGIN **********************')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=7)
    X = lowVarianceFilter(X, X_train, varianceTreshold)
    print('-----------------------------------------------------------------------------')
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=7)
    # X, Y = bestFeaturesFiltering(X, X_train, y_train, percentage)
    print('*************************** Feature Selection -  END ************************')
    return X, Y


def lowVarianceFilter(X, X_train, t):
    # Perform feature selection using a variance threshold
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=t)
    # sel.fit(X_train)
    sel.fit(X_train)
    print("Feature filtering", sel.get_support())
    print("Remaining features:", list(X.columns[sel.get_support()]))
    print("Filtered features:", list(X.columns[~sel.get_support()]))
    columns_list = list(X.columns[sel.get_support()])
    # Transform (remove low variance features)
    # X_train = sel.transform(X_train)
    # X_test = sel.transform(X_test)
    X = sel.transform(X)
    X = pd.DataFrame(X, columns=columns_list)
    return X


def bestFeaturesFiltering(X, X_train, y_train, perc):
    bestPerc = SelectPercentile(score_func=f_regression, percentile=perc)
    # kbest.fit(X_train, y_train)
    bestPerc.fit(X_train, y_train)

    print("Feature selection", bestPerc.get_support())
    print("Feature scores", bestPerc.scores_)
    print("Selected features:", list(X.columns[bestPerc.get_support()]))
    print("Removed features:", list(X.columns[~bestPerc.get_support()]))
    columns_list = list(X.columns[bestPerc.get_support()])
    X = bestPerc.transform(X)
    X = pd.DataFrame(X, columns=columns_list)
    return X


# Pre-Processing function: PCA (Principal Components Analysis)
def principalComponentAnalysis(X_train, X_test, components):
    pca = decomposition.PCA(n_components=components)
    pca.fit(X_train)
    # round the explained variance
    u = [round(i * 100, 2) for i in pca.explained_variance_ratio_]
    print("------------------------PRINCIPAL COMPONENT ANALYSIS-----------------")
    print("pca.explained_variance_ratio: ", u)
    print("sum pca explained_variance_ratio ", np.sum(u))
    X_train_t = pca.transform(X_train)
    X_test_t = pca.transform(X_test)
    #print(pd.DataFrame(pca.components_, columns=X_train.columns))
    #plt.plot(pd.DataFrame(pca.components_, columns=X_train.columns))
    # plt.plot(pca.components_)
    return X_train_t, X_test_t

def L2_reg(X):
    x_norm1 = np.linalg.norm(X, ord=2)
    x_normalized = X / x_norm1
    return x_normalized

if __name__ == '__main__':
    np.random.seed(123456)
    df = pd.read_csv('ImportDataset.csv')
    X_train, X_test, y_train, y_test = preprocessing(df, True, 0.02, 10, True, 120, True)


