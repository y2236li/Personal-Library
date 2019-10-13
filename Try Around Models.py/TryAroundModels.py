import re
import six
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import check_X_y, check_array
from sklearn.preprocessing import LabelBinarizer, normalize
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.svm import LinearSVC 
from sklearn.base import BaseEstimator, ClassifierMixin

from scipy.sparse import issparse
from scipy import sparse

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

from keras import backend as K

from tqdm import tqdm
from abc import ABCMeta




class NBSVM(six.with_metaclass(ABCMeta, BaseEstimator, ClassifierMixin)):

    def __init__(self, alpha=1.0, C=1.0, max_iter=10000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.C = C
        self.svm_ = [] # fuggly

    def fit(self, X, y):
        X, y = check_X_y(X, y, 'csr')
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # so we don't have to cast X to floating point
        Y = Y.astype(np.float64)

        # Count raw events from data
        n_effective_classes = Y.shape[1]
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.ratios_ = np.full((n_effective_classes, n_features), self.alpha,
                                 dtype=np.float64)
        self._compute_ratios(X, Y)

        # flugglyness
        for i in range(n_effective_classes):
            X_i = X.multiply(self.ratios_[i])
            svm = LinearSVC(C=self.C, max_iter=self.max_iter)
            Y_i = Y[:,i]
            svm.fit(X_i, Y_i)
            self.svm_.append(svm) 

        return self

    def predict(self, X):
        n_effective_classes = self.class_count_.shape[0]
        n_examples = X.shape[0]

        D = np.zeros((n_effective_classes, n_examples))

        for i in range(n_effective_classes):
            X_i = X.multiply(self.ratios_[i])
            D[i] = self.svm_[i].decision_function(X_i)
        
        return self.classes_[np.argmax(D, axis=0)]
        
    def _compute_ratios(self, X, Y):
        """Count feature occurrences and compute ratios."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")

        self.ratios_ += safe_sparse_dot(Y.T, X)  # ratio + feature_occurrance_c
        normalize(self.ratios_, norm='l1', axis=1, copy=False)
        row_calc = lambda r: np.log(np.divide(r, (1 - r)))
        self.ratios_ = np.apply_along_axis(row_calc, axis=1, arr=self.ratios_)
        check_array(self.ratios_)
        self.ratios_ = sparse.csr_matrix(self.ratios_)

        #p_c /= np.linalg.norm(p_c, ord=1)
        #ratios[c] = np.log(p_c / (1 - p_c))


def f1_class(pred, truth, class_val):
    n = len(truth)

    truth_class = 0
    pred_class = 0
    tp = 0

    for ii in range(0, n):
        if truth[ii] == class_val:
            truth_class += 1
            if truth[ii] == pred[ii]:
                tp += 1
                pred_class += 1
                continue;
        if pred[ii] == class_val:
            pred_class += 1

    precision = tp / float(pred_class)
    recall = tp / float(truth_class)

    return (2.0 * precision * recall) / (precision + recall)


def semeval_senti_f1(pred, truth, pos=2, neg=0): 

    f1_pos = f1_class(pred, truth, pos)
    f1_neg = f1_class(pred, truth, neg)

    return (f1_pos + f1_neg) / 2.0;


def main(train_file, test_file, ngram=(1, 3)):
    print('loading...')
    train = pd.read_csv(train_file, delimiter='\t', encoding='utf-8', header=0,
                        names=['text', 'label'])

    # to shuffle:
    #train.iloc[np.random.permutation(len(df))]

    test = pd.read_csv(test_file, delimiter='\t', encoding='utf-8', header=0,
                        names=['text', 'label'])

    print('vectorizing...')
    vect = CountVectorizer()
    classifier = NBSVM()

    # create pipeline
    clf = Pipeline([('vect', vect), ('nbsvm', classifier)])
    params = {
        'vect__token_pattern': r"\S+",
        'vect__ngram_range': ngram, 
        'vect__binary': True
    }
    clf.set_params(**params)

    #X_train = vect.fit_transform(train['text'])
    #X_test = vect.transform(test['text'])

    print('fitting...')
    clf.fit(train['text'], train['label'])

    print('classifying...')
    pred = clf.predict(test['text'])
   
    print('testing...')
    acc = accuracy_score(test['label'], pred)
    f1 = semeval_senti_f1(pred, test['label'])
    print('NBSVM: acc=%f, f1=%f' % (acc, f1))





def TryAroundModel_LG(X_train, X_test, Y_train, Y_test):
    logistic_model = LogisticRegression(solver='lbfgs')
    logistic_model.fit(X_train, Y_train)
    LG_accuracy = accuracy_score(logistic_model.predict(X_test), Y_test)
    print("Logistic Regression -- Accuracy: ", LG_accuracy)
    
    return ("Logistic Regression", LG_accuracy)

# TryAroundModel_LG(X_train, X_test, Y_train, Y_test)


def TryAroundModel_NB(X_train, X_test, Y_train, Y_test):
    NB_model = MultinomialNB(alpha=7.4)
    NB_model.fit(X_train, Y_train)
    NB_accuracy = accuracy_score(NB_model.predict(X_test), Y_test)
    
    print("Multinomial Naive Bayes -- Accuracy: ", NB_accuracy)

    return ("Multinomial Naive Bayes", NB_accuracy)

# TryAroundModel_NB(X_train, X_test, Y_train, Y_test)


def TryAroundModel_RF(X_train, X_test, Y_train, Y_test):
    RF_model = RandomForestClassifier(n_estimators= 32)
    RF_model = RF_model.fit(X_train, Y_train)
    RF_accuracy = accuracy_score(RF_model.predict(X_test), Y_test)
    
    print("Random Forest -- Accuracy: ", RF_accuracy)

    return ("Random Forest", RF_accuracy)

# TryAroundModel_RF(X_train, X_test, Y_train, Y_test)


def TryAroundModel_GBM(X_train, X_test, Y_train, Y_test):
    GBM_model = GradientBoostingClassifier(n_estimators= 10, n_iter_no_change = 3)
    GBM_model = GBM_model.fit(X_train, Y_train)
    GBM_accuracy = accuracy_score(GBM_model.predict(X_test), Y_test)
    
    print("Gradient Boosting Machine -- Accuracy: ", GBM_accuracy)
    
    return ("Gradient Boosting Machine", GBM_accuracy)

# TryAroundModel_GBM(X_train, X_test, Y_train, Y_test)

def TryAroundModel_NBSVM(X_train, X_test, Y_train, Y_test):
    NB_SVM_model = NBSVM()
    NB_SVM_model = NB_SVM_model.fit(X_train, Y_train)
    NB_SVM_accuracy = accuracy_score(NB_SVM_model.predict(X_test), Y_test)
    
    print("Naive Bayes SVM -- Accuracy: ", NB_SVM_accuracy)
    
    return ("Naive Bayes SVM", NB_SVM_accuracy)

# TryAroundModel_NBSVM(X_train, X_test, Y_train, Y_test)


def TryAroundModel_MPLNN(X_train, X_test, Y_train, Y_test):
    
    MPLNN_model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape = (X_train.shape[1],)),
        tf.keras.layers.Dense(2800, activation="relu"),
        tf.keras.layers.Dropout(0),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0),
        tf.keras.layers.Dense(2, activation = "softmax")
    ])

    MPLNN_model.compile(optimizer = 'rmsprop', loss = 'sparse_categorical_crossentropy')
    MPLNN_model.fit(X_train, Y_train)
    y_predict = MPLNN_model.predict_classes(X_test)
    MPLNN_accuracy = accuracy_score(y_predict, Y_test)
    
    print("Multilayer Perceptron Neural Network(MLP) -- Accuracy: ", MPLNN_accuracy)
    
    return ("Multilayer Perceptron Neural Network(MLP)", MPLNN_accuracy)

# TryAroundModel_MPLNN(X_train, X_test, Y_train, Y_test)


def TryAroundModel_LSTM(X_raw_text_train, X_raw_text_test, Y_train, Y_test):
    max_features = 10000
    max_len = 200
    tokenizer = Tokenizer(nb_words = max_features)
    tokenizer.fit_on_texts(X_raw_text_train)
    sequences_train = tokenizer.texts_to_sequences(X_raw_text_train)
    sequences_test = tokenizer.texts_to_sequences(X_raw_text_test)

    X_train = sequence.pad_sequences(sequences_train, maxlen=max_len)
    X_test = sequence.pad_sequences(sequences_test, maxlen=max_len)

    Y_train = tf.keras.utils.to_categorical(Y_train)
    Y_test = tf.keras.utils.to_categorical(Y_test)


    LSTM_model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(input_dim = max_features, output_dim = 200, mask_zero = True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(300,
                                 dropout= 0.2, recurrent_dropout=0.2),
            tf.keras.layers.Dense(2, activation = "softmax", input_shape = (32,300))
    ])


    LSTM_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['categorical_accuracy'])

    
    LSTM_model.fit(X_train, Y_train, batch_size=32, epochs=1)
    score, acc = LSTM_model.evaluate(X_test, Y_test,
                                batch_size=32)
    
    print("LSTM Neural Network -- Accuracy: ", acc)
    
    return ("LSTM Neural Network", acc)
    
    
# TryAroundModel_LSTM(X_raw_text_train, X_raw_text_test, Y_train, Y_test)

def TryAroundModel_FB_LSTM(X_raw_text_train, X_raw_text_test, Y_train, Y_test):
    max_features = 10000
    max_len = 200
    tokenizer = Tokenizer(nb_words = max_features)
    tokenizer.fit_on_texts(X_raw_text_train)
    sequences_train = tokenizer.texts_to_sequences(X_raw_text_train)
    sequences_test = tokenizer.texts_to_sequences(X_raw_text_test)

    X_train = sequence.pad_sequences(sequences_train, maxlen=max_len)
    X_test = sequence.pad_sequences(sequences_test, maxlen=max_len)

    Y_train = tf.keras.utils.to_categorical(Y_train)
    Y_test = tf.keras.utils.to_categorical(Y_test)


    forward_LSTM = tf.keras.layers.LSTM(300, dropout= 0.2, recurrent_dropout=0.2, return_sequences=True)
    backward_LSTM = tf.keras.layers.LSTM(300, dropout= 0.2, recurrent_dropout=0.2, return_sequences=True, go_backwards = True)


    LSTM_model = tf.keras.models.Sequential()
    LSTM_model.add(tf.keras.layers.Embedding(input_dim = max_features, output_dim = 200, mask_zero = True))
    LSTM_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300, return_sequences=True), input_shape=(32, 200)))
    LSTM_model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(300)))
    LSTM_model.add(tf.keras.layers.Dense(2, activation = "softmax"))


    LSTM_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['categorical_accuracy'])


    LSTM_model.fit(X_train, Y_train, batch_size=32, epochs=1)
    
    
    score, acc = LSTM_model.evaluate(X_test, Y_test,
                                batch_size=32)
    
    
    print("Forward and Backward LSTM Neural Netword -- Accuracy: ", acc)
    
    return ("Forward and Backward LSTM Neural Netword", acc)
    
    
# TryAroundModel_FB_LSTM(X_raw_text_train, X_raw_text_test, Y_train, Y_test)


def TryAroundModel_CNN(X_raw_text_train, X_raw_text_test, Y_train, Y_test):
    max_features = 10000
    max_len = 200
    tokenizer = Tokenizer(nb_words = max_features)
    tokenizer.fit_on_texts(X_raw_text_train)
    sequences_train = tokenizer.texts_to_sequences(X_raw_text_train)
    sequences_test = tokenizer.texts_to_sequences(X_raw_text_test)

    X_train = sequence.pad_sequences(sequences_train, maxlen=max_len)
    X_test = sequence.pad_sequences(sequences_test, maxlen=max_len)

    Y_train = tf.keras.utils.to_categorical(Y_train)
    Y_test = tf.keras.utils.to_categorical(Y_test)

    num_filters = 168
    CNN_model = tf.keras.models.Sequential()
    CNN_model.add(tf.keras.layers.Embedding(input_dim = max_features, output_dim = 200))
    CNN_model.add(tf.keras.layers.Convolution1D(filters = num_filters, kernel_size = 20, activation = 'relu'))
    
    def Max1D(X):
        return K.max(X, axis = 1)
    
    CNN_model.add(tf.keras.layers.Lambda(Max1D, output_shape = (num_filters, )))
    CNN_model.add(tf.keras.layers.Dense(120, activation = 'relu'))
    CNN_model.add(tf.keras.layers.Dropout(0.2))
    CNN_model.add(tf.keras.layers.Dense(2, activation = 'softmax'))

    CNN_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['categorical_accuracy'])
    
    CNN_model.fit(X_train, Y_train, batch_size=32, epochs=1)
    
    score, acc = CNN_model.evaluate(X_test, Y_test,
                            batch_size=32)
    
    print("Convolutional Neural Network -- Accuracy: ", acc)
    
    
    return ("Convolutional Neural Network", acc)
    

# TryAroundModel_CNN(X_raw_text_train, X_raw_text_test, Y_train, Y_test)



def TryAroundModel(X_train, X_test, Y_train, Y_test, X_raw_text_train = None, X_raw_text_test = None, Models = None):
    if not Models:
        Models = []
        for i in np.nonzero([re.match("TryAroundModel", x) for x in globals().keys()])[0]:
            Models.append(list(globals().keys())[i])
    
    accuracy_list = []
    processed_arg = [X_train, X_test, Y_train, Y_test]
    
    for m in Models:
        if m == "TryAroundModel_LG":
            accuracy_list.append(TryAroundModel_LG(*processed_arg))
        elif m == "TryAroundModel_NB":
            accuracy_list.append(TryAroundModel_NB(*processed_arg))
        elif m == "TryAroundModel_NBSVM":
            accuracy_list.append(TryAroundModel_NBSVM(*processed_arg))
        elif m == "TryAroundModel_RF":
            accuracy_list.append(TryAroundModel_RF(*processed_arg))
        elif m == "TryAroundModel_GBM":
            accuracy_list.append(TryAroundModel_GBM(*processed_arg))
        elif m == "TryAroundModel_MPLNN":
            accuracy_list.append(TryAroundModel_MPLNN(*processed_arg))

        if X_raw_text_train is not None and X_raw_text_test is not None:
            raw_arg = [X_raw_text_train, X_raw_text_test, Y_train, Y_test]
            if m == "TryAroundModel_CNN":
                accuracy_list.append(TryAroundModel_CNN(*raw_arg))
            elif m == "TryAroundModel_LSTM":
                accuracy_list.append(TryAroundModel_LSTM(*raw_arg))
            elif m == "TryAroundModel_FB_LSTM":
                accuracy_list.append(TryAroundModel_FB_LSTM(*raw_arg))
        
    return sorted(accuracy_list, key = lambda x: x[1], reverse = True)
