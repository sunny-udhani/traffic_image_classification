
# coding: utf-8

# ### Traffic congestion seems to be at an all-time high. Machine Learning methods must be developed to help solve traffic problems. In this assignment, you will analyze features extracted from traffic imagesdepicting different objects to determine their type as one of 11classes, noted by integers 1-11: car, suv, small_truck, medium_truck, large_truck, pedestrian, bus, van, people, bicycle, and motorcycle. The object classes are heavily imbalanced. For example, the training data contains 10375 cars but only 3 bicycles and 0 people. Classes in the test data are similarly distributed.
# 
# * Use/implement a feature selection/reduction technique. Some sort of feature selection or dimensionality reduction must be included in your final problem solution.
# * Experiment with various classification models.
# * Think about dealing with imbalanced data.
# * F1 Scoring Metric

# In[15]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn.manifold as manifold
from numpy import genfromtxt
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import Counter, defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from MulticoreTSNE import MulticoreTSNE as TSNE


# In[16]:


# utility to read data from a fall path given
def give_me_some_data(path_to_file, headers = None, separator = ',',float_precision = None):
    if(path_to_file):
        return pd.read_csv(
            filepath_or_buffer=path_to_file, 
            header=headers, 
            sep=separator,
            float_precision=float_precision
        )


# In[17]:


# utility to get a Dimensionality Reduction method based on input params
def give_me_a_dr(dr_tech = "PCA", n_components = 40, n_neighbours = 100):
    switcher = {
        "PCA" : PCA(n_components=n_components),
        "TruncatedSVD": TruncatedSVD(n_components=n_components),
        "KBest": SelectKBest(chi2, k=n_components),
        "LLE-Standard": LocallyLinearEmbedding(n_neighbors=n_neighbours,
                                             n_components=n_components, method = 'standard'),
        "LLE-Hessian": LocallyLinearEmbedding(n_neighbors=n_neighbours,
                                             n_components=n_components, method = 'hessian'),
        "spectral_embedding": SpectralEmbedding(n_components=n_components, random_state=0,
                                      eigen_solver="arpack"),
#         Takes too long!!!
#         "tsne": TSNE(n_components=n_components, init='pca', random_state=0)
    }
    return switcher[dr_tech]


# In[18]:


# utility to get a Classification method based on input params
def give_me_a_classifier(classifier = "SVC", neighbours = None):
    switcher = {
        "SVC": SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape='ovr', degree=3, gamma=2, kernel='linear',
                  max_iter=-1, probability=False, random_state=42, shrinking=True,
                  tol=0.001, verbose=False),
        "SGD": SGDClassifier(random_state=42, max_iter=1000, tol=1e-3),
        "KNN": KNeighborsClassifier(n_neighbors=neighbours),
        "RF": RandomForestClassifier(n_estimators=neighbours, class_weight="balanced"),
        "NN": MLPClassifier(hidden_layer_sizes=(neighbours,neighbours,neighbours),
                            max_iter=2000),
        "GP":GaussianProcessClassifier(1.0 * RBF(1.0)),
        "boost": AdaBoostClassifier(
                                    DecisionTreeClassifier(min_samples_split=4,
                                                           class_weight="balanced",
                                                           random_state=21),
                                    n_estimators=300,
                                    learning_rate=1.5,
                                    algorithm="SAMME"),
        "GNB": GaussianNB(),
        "BNB": BernoulliNB(),
        "EXT": ExtraTreesClassifier(n_estimators=neighbours,
                                    min_samples_split=4,
#                                     max_depth=15,
                                    random_state=21,
                                    class_weight="balanced"),
    }
    return switcher[classifier]


# In[19]:


# utility to get a Pipeline containing things based on input params
def give_me_a_pipeline(dr = None, classifier = None):
    return  Pipeline(steps=[("DR", dr), ('classify', classifier)])


# In[20]:


# utility to fix imbalance of training dataset
def fix_my_data_imbalance_pls(training_data, training_classes):
    nm = SMOTE(random_state=21,k_neighbors=2,kind="regular")
    training_data = np.array(training_data)
    
#     try to normalize the values, eventually decided not needed
#     training_data = normalize(training_data)
#     print(training_data[:5])

    training_classes = np.array(training_classes)
    
#     Try out AdaSYN for imbalance fix, performance didn't improve
#     ada = ADASYN(random_state=21, n_neighbors=2)
#     X_res, y_res = ada.fit_sample(training_data, training_classes)

#     Try out RandomOverSampler for imbalance fix, no performance improvement
#     ro = RandomOverSampler(random_state=21, n_neighbors=2)
#     X_res, y_res = ro.fit_sample(training_data, training_classes)
    
    X_res, y_res = nm.fit_sample(training_data, training_classes)
    
#   Plot the graph for fixed balanced dataset, ready for training  
#     labels, values = zip(*Counter(y_res).most_common())
#     indexes = np.arange(len(labels))
#     width = 1
#     plt.bar(indexes, values, width)
#     plt.xticks(indexes + width * 0.5, labels)
#     plt.show()
    
    return X_res, y_res


# ### The input to your analysis will not be the images themselves, but rather features extracted from the images. An image can be can be described by many different types of features. In the training and test datasets, images are described as 887-dimensional vectors, composed by concatenating the following features:- 
# * 512 Histogram of Oriented Gradients (HOG) features
# * 256 Normalized Color Histogram (Hist) features
# - 64 Local Binary Pattern (LBP) features 
# - 48 Color gradient (RGB) features
# - 7 Depth of Field (DF) features

# In[36]:


# read in the dataset - training set, labels and test set
training_data = give_me_some_data(
    path_to_file='../data/train.dat', 
    headers=None, 
    separator=' ',
    float_precision='high'
)

test_data = give_me_some_data(
    path_to_file='../data/test.dat', 
    headers=None, 
    separator=' ',
    float_precision='high' 
)

training_labels = give_me_some_data(
    path_to_file='../data/train.labels', 
    headers=None, 
)

# choose only invariant features
training_data = training_data[training_data.columns[832:880]]
test_data = test_data[test_data.columns[832:880]]

# test_data

# Print out the number of uniques values in each feature of the dataset
# df.to_csv("../data/aaj.csv", sep='\t', encoding='utf-8')
# df.nunique().to_csv("../data/unique_vals.csv", sep='\t', encoding='utf-8')
# cols = df.columns
# for col in cols:
#     df[col] = df[col].astype(float) 
# print(df)

# extract the vectors from the Pandas data file
# X = df.iloc[:,1:].values
# df
# my_data = np.loadtxt('../data/train.dat', delimiter=' ',dtype = np.float64)
# np.set_printoptions(precision = 20)
# my_data
# df1 = pd.DataFrame(my_data)
# df1

# standardise the data
# X_std = StandardScaler().fit_transform(X)


# ### Since the dataset is imbalanced the scoring function will be the F1-score instead of Accuracy.

# In[37]:


# Fix unbalanced class distributuion before performing any analysis
training_data,training_labels = fix_my_data_imbalance_pls(training_data=training_data,
                                                          training_classes=training_labels)


# In[38]:


# Shuffle data from training set
X_train, X_test, y_train, y_test = train_test_split(
    training_data,
    training_labels,
    test_size=0.2,
    shuffle=True,
    random_state=21,
)


# In[39]:


# Evaluate and cross-validate different hyper-parameters and estimators on training set

# Create a pipeline from the type of techniques needed and use trained pipeline for prediction
compute_pipeline = give_me_a_pipeline(give_me_a_dr(dr_tech="KBest",n_components=38),
                                      give_me_a_classifier(classifier="EXT",neighbours=500))

compute_pipeline.fit(X_train, y_train)

# try out Fast-TSNE, LLE for DR, can't use pipeline as both lack separate fit and transform methods
# tsne=LocallyLinearEmbedding(n_neighbors=100,
#                           n_components=10,
#                           method = 'standard')
# X_tsne = tsne.fit_transform(X_train,y_train)
# RF = RandomForestClassifier(n_estimators=100)
# RF.fit(X_tsne,y_train)

y_pred = compute_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))


# In[42]:


# Join the training dataset to train the model and predict output for test set

X_train = np.vstack((X_train,X_test))
y_train = np.append(y_train,y_test)

compute_pipeline = give_me_a_pipeline(give_me_a_dr(dr_tech="KBest", n_components=46),
                                      give_me_a_classifier(classifier="EXT",neighbours=500))

compute_pipeline.fit(X_train, y_train)
y_pred = compute_pipeline.predict(test_data)


# In[43]:


# send output to out.dat
with open('out.dat', 'w') as f:
    for i in range(test_data.shape[0]):
        f.write("%s\n" % (y_pred[i]))

