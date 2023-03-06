import numpy as np
import pandas as pd
import random

import docx

from collections import defaultdict


def confusion_matrix(y_true, y_pred):
    """
    Computes the confusion matrix.

    Parameters:
    y_true (array-like): The true labels.
    y_pred (array-like): The predicted labels.

    Returns:
    array-like: 2D array of TP, FP, TN, FN
    """
    # Find unique labels in y_true
    labels = y_true.unique().tolist()

    # Initialize a matrix of zeros
    cm = [[0 for _ in range(len(labels))] for _ in range(len(labels))]

    # Fill in the matrix by counting occurrences of each label pair
    for i in range(len(y_true)):
        true_index = labels.index(y_true[i])
        pred_index = labels.index(y_pred[i])
        cm[true_index][pred_index] += 1

    return labels, cm


def accuracy_score(y_true, y_pred):
    """
    Computes the accuracy score.
    Parameters:
    y_true (array-like): The true labels.
    y_pred (array-like): The predicted labels.

    Returns:
    float: The accuracy score.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    # Compute the number of correctly classified samples
    n_correct = np.sum(y_true == y_pred)

    # Compute the accuracy score
    accuracy = n_correct / len(y_true)

    return accuracy


def precision_score(y_true, y_pred, average):
    """
    Computes the precision score.

    Parameters:
    y_true (array-like): The true labels.
    y_pred (array-like): The predicted labels.
    average (str): The type of averaging to perform. Possible values are
        'binary', 'micro', 'macro', 'weighted', and 'samples'.

    Returns:
    float: The precision score.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if average not in ('binary', 'micro', 'macro', 'weighted', 'samples'):
        raise ValueError(
            "average must be one of 'binary', 'micro', 'macro', 'weighted', or 'samples'")

    # Compute the confusion matrix
    _, conf_mat = confusion_matrix(y_true, y_pred)

    # Compute the precision scores for each class
    tp = np.diag(conf_mat)
    fp = np.sum(conf_mat, axis=0) - tp
    precision = tp / (tp + fp)

    # Compute the average precision score
    if average == 'binary':
        precision = precision[1]
    elif average == 'micro':
        precision = np.sum(tp) / np.sum(tp + fp)
    elif average == 'macro':
        precision = np.mean(precision)
    elif average == 'weighted':
        weights = np.sum(conf_mat, axis=1)
        precision = np.average(precision, weights=weights)
    elif average == 'samples':
        precision = precision.mean()

    return precision


def recall_score(y_true, y_pred, average='binary'):
    """
    Computes the recall score.

    Parameters:
    y_true (array-like): The true labels.
    y_pred (array-like): The predicted labels.
    average (str): The type of averaging to perform. Possible values are
        'binary', 'micro', 'macro', 'weighted', and 'samples'.

    Returns:
    float: The recall score.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if average not in ('binary', 'micro', 'macro', 'weighted', 'samples'):
        raise ValueError(
            "average must be one of 'binary', 'micro', 'macro', 'weighted', or 'samples'")

    # Compute the confusion matrix
    _, conf_mat = confusion_matrix(y_true, y_pred)

    # Compute the recall scores for each class
    tp = np.diag(conf_mat)
    fn = np.sum(conf_mat, axis=1) - tp
    recall = tp / (tp + fn)

    # Compute the average recall score
    if average == 'binary':
        recall = recall[1]
    elif average == 'micro':
        recall = np.sum(tp) / np.sum(tp + fn)
    elif average == 'macro':
        recall = np.mean(recall)
    elif average == 'weighted':
        weights = np.sum(conf_mat, axis=1)
        recall = np.average(recall, weights=weights)
    elif average == 'samples':
        recall = recall.mean()

    return recall


def cv(k, data_split):
    """Perform Cross Validation

    Args:
        k (int): Number of folds
        data_split (list): List of folds

    Returns:
        NaiveBayes: Model with best accuracy of cross validations.
    """

    results = []
    best_model = None

    for i in range(k):
        cv = []
        for j in range(k):
            if j != i:
                cv += data_split[j]

        temp_df = pd.DataFrame(cv)

        x = temp_df.drop([temp_df.columns[-1]], axis=1)
        y = temp_df[temp_df.columns[-1]]

        test_df = pd.DataFrame(data_split[i])

        model = NaiveBayes()
        model.fit(x, y)
        y_pred = model.predict(test_df.drop([test_df.columns[-1]], axis=1))

        results.append(accuracy_score(test_df[test_df.columns[-1]], y_pred))
        print("\nK = {} Accuracy: {}".format(
            i+1, results[-1]))

        if not best_model:
            best_model = (model, results[0])
        else:
            if best_model[1] > results[-1]:
                best_model = (model, results[-1])

    print("\nAverage Accuracy: {}".format(sum(results)/k))
    return best_model[0]


def stratified_k_fold_cross_validation(df, k):
    """
    Perform stratified k-fold cross-validation on a given dataset.

    Parameters:
        X (array-like): The feature data.
        y (array-like): Label data.
        k (int): Number of folds.
        model (object): Model object.

    Returns:
        NaiveBayes: Model with best accuracy of cross validations.
    """

    df_copy = df.copy()

    # dict to hold data by class label
    data_by_label = defaultdict(list)
    for idx, row in df_copy.iterrows():
        data_by_label[df_copy.columns[-1]].append(df_copy.iloc[idx])

    # shuffle data for each label
    for label in data_by_label.keys():
        random.shuffle(data_by_label[label])

    data_split = [[] for _ in range(k)]

    # add data to each fold
    for label, data in data_by_label.items():
        # calculate number of data in a fold
        fold_size = len(data) // k
        for i in range(k):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size
            if i == k-1:
                end_idx = len(data)
            data_split[i] += data[start_idx:end_idx]

    return cv(k, data_split)


def k_fold_cross_validation(df, k):
    """
    Perform k-fold cross-validation on a given dataset.

    Parameters:
        X (array-like): The feature data.
        y (array-like): Label data.
        k (int): Number of folds.
        model (object): Model object.

    Returns:
        NaiveBayes: Model with best accuracy of cross validations.
    """

    df_copy = df.copy()

    # calculate number of data in a fold
    fold_size = df_copy.shape[0] // k

    data_split = [[] for _ in range(k)]

    # add data to each fold
    for i in range(k):
        while len(data_split[i]) < fold_size:
            idx = df_copy.index[random.randrange(df_copy.shape[0])]
            data_split[i].append(df_copy.loc[idx].values.tolist())
            df_copy.drop(idx, inplace=True)

    return cv(k, data_split)


class NaiveBayes:
    """
    A Naive Bayes classifier for classification problems.
    """

    def __init__(self):
        """
        Initializes the NaiveBayes class.
        """
        self.attributes = []
        self.likelihoods = {}
        self.class_prior_probs = {}
        self.pred_prior_probs = {}

        self.x_train = None
        self.y_train = None
        self.train_size = 0
        self.num_feats = 0

    def fit(self, x, y):
        """
        Trains the Naive Bayes classifier on the input data.

        Parameters:
            x (pandas.DataFrame): The feature data.
            y (pandas.Series): The label data.
        """
        self.x_train = x
        self.y_train = y
        self.train_size = x.shape[0]
        self.num_feats = x.shape[1]
        self.attributes = x.columns.tolist()

        for attribute in self.attributes:
            self.likelihoods[attribute] = {}
            self.pred_prior_probs[attribute] = {}

            for attribute_val in np.unique(self.x_train[attribute]):
                self.pred_prior_probs[attribute].update({attribute_val: 0})

                for out in np.unique(self.y_train):
                    self.likelihoods[attribute].update(
                        {attribute_val + '_' + out: 0})
                    self.class_prior_probs.update({out: 0})

        self._calc_class_prior()
        self._calc_likelihoods()
        self._calc_predictor_prior()

    def _calc_class_prior(self):
        """
        Calculates the class prior probabilities for the Naive Bayes classifier.
        """
        for out in np.unique(self.y_train):
            out_count = sum(self.y_train == out)
            self.class_prior_probs[out] = out_count / self.train_size

    def _calc_likelihoods(self):
        """
        Calculates the likelihoods of the features given the class labels for the Naive Bayes classifier.
        """
        for attribute in self.attributes:

            for out in np.unique(self.y_train):
                out_count = sum(self.y_train == out)
                attribute_likelihood = self.x_train[attribute][self.y_train[self.y_train == out].index.values.tolist(
                )].value_counts().to_dict()

                for attribute_val, count in attribute_likelihood.items():
                    self.likelihoods[attribute][attribute_val +
                                                '_' + out] = count/out_count

    def _calc_predictor_prior(self):
        """
        Calculates the prior probabilities of the features for the Naive Bayes classifier.
        """
        for attribute in self.attributes:
            attribute_vals = self.x_train[attribute].value_counts().to_dict()

            for attribute_val, count in attribute_vals.items():
                self.pred_prior_probs[attribute][attribute_val] = count / \
                    self.train_size

    def predict(self, x):
        """
        Predict the class labels for the given input data

        Parameters:
            x (pandas.DataFrame): The feature data.

        Returns:
        np.array(): A numpy array containing the predicted labels.
        """
        results = []
        x = np.array(x)

        for data in x:
            prob_out = {}
            for out in np.unique(self.y_train):
                prior = self.class_prior_probs[out]
                likelihood = 1
                evidence = 1

                for attribute, attribute_val in zip(self.attributes, data):
                    likelihood *= self.likelihoods[attribute][attribute_val + '_' + out]
                    evidence *= self.pred_prior_probs[attribute][attribute_val]

                posterior = (likelihood * prior) / (evidence)

                prob_out[out] = posterior

            result = max(prob_out, key=lambda x: prob_out[x])
            results.append(result)

        return np.array(results)


# def read_file(f_name):
#     """Read data from txt file

#     Args:
#         f_name (str): txt file name

#     Returns:
#         pandas.DataFrame: Loaded dataset
#     """
#     df = pd.read_csv(f_name, delimiter=',', header=None)
#     return df

def read_file(f_name):
    """
    Read and parse file from *.Docx extension

    Parameters:
        f_name (str): The file name

    Returns:
        pandas.DataFrame: Training or testing data
    """
    doc = docx.Document(f_name)

    txt = []

    for para in doc.paragraphs[:-1]:
        txt.append(para.text)

    df = pd.DataFrame({'text': txt})
    df['values'] = df['text'].str.split(',')
    new_df = pd.DataFrame(df['values'].tolist())
    df = pd.concat([df, new_df], axis=1)
    df = df.drop(['text', 'values'], axis=1)

    return df


while True:
    try:
        usr_in = None
        while True:
            usr_in = input(
                '\nEnter 1 to train\nEnter 2 to classify\nEnter 3 to test accuracy\nEnter 4 to k-Fold cv\nEnter 5 to stratified k-Fold cv\nEnter 6 to exit\n')

            print(f'You entered: {usr_in}')

            if usr_in < '1' or usr_in > '6':
                print("Press appropiate number")

            else:
                break

        if usr_in == '1':
            meta_name = input("\nEnter meta filename: ")
            f_name = input("\nEnter training filename: ")

            df = read_file(f_name)

            x = df.drop([df.columns[-1]], axis=1)
            y = df[df.columns[-1]]

            nb_clf = NaiveBayes()
            nb_clf.fit(x, y)
            print('\nModel trained on data...\n')

        elif usr_in == '2':
            data = input(
                "Enter data separated by commas in the order which is given in the other files: ")

            # Split the data by commas and store it in a list
            data_list = data.split(",")
            column_list = nb_clf.attributes
            # Create a dataframe from the data and column lists
            df = pd.DataFrame(data=list([data_list]), columns=column_list)
            print("The prediction for the input is: "+nb_clf.predict(df)[0])

        elif usr_in == '3':
            f_name = input("\nEnter testing filename: ")
            df = read_file(f_name)

            x = df.drop([df.columns[-1]], axis=1)
            y = df[df.columns[-1]]

            y_pred = nb_clf.predict(x)

            print("\nAccuracy: {}".format(accuracy_score(y, y_pred)))
            print("Precision: {}".format(precision_score(y, y_pred, 'weighted')))
            print("Recall: {}".format(
                recall_score(y, y_pred, average='weighted')))
            print("Confusion Matrix: \n")
            labels, cm = confusion_matrix(y, y_pred)

            print("{:<8}".format(''), *
                  ["{:<8}".format(label) for label in labels])

            for i in range(len(labels)):
                print("{:<8}".format(labels[i]), *
                      ["{:<8}".format(val) for val in cm[i]])

            print()

        elif usr_in == '4' or usr_in == '5':
            meta_name = input("\nEnter meta filename: ")
            f_name = input("\nEnter training filename: ")
            k = int(input("\nEnter value for K: "))

            if k < 2:
                raise ValueError("K should be greater than one.")

            df = read_file(f_name)
            nb_clf = k_fold_cross_validation(
                df, k) if usr_in == '4' else stratified_k_fold_cross_validation(df, k)

        else:
            print('Terminated program')
            break

    except(FileNotFoundError):
        print("ERROR: FILE NOT FOUND")
    except(NameError):
        print("ERROR: NO TRAINED MODEL FOUND")
