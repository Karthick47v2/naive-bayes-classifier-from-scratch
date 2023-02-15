from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
import docx
import io


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


def read_file(f_name):
    """
    Read and parse file from *.Docx extension

    Args:
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
            print(
                '--------------------------------------------------------------------------------')
            usr_in = input(
                '1. Train Classifier\n2. Test Classifier\n3. Exit\nSelect option: ')

            if usr_in != '1' and usr_in != '2' and usr_in != '3':
                print("Press 1 or 2")

            else:
                break

        if usr_in == '1':
            f_name = input("\nEnter file name of training data: ")
            meta_name = input("\nEnter file name of meta data: ")

            df = read_file(f_name)

            x = df.drop([df.columns[-1]], axis=1)
            y = df[df.columns[-1]]

            nb_clf = NaiveBayes()
            nb_clf.fit(x, y)

            print('\nModel trained on data...\n')

        elif usr_in == '2':
            f_name = input("\nEnter file name of test data: ")
            df = read_file(f_name)

            x = df.drop([df.columns[-1]], axis=1)
            y = df[df.columns[-1]]

            f_name = input('\nEnter output file name: ')

            y_pred = nb_clf.predict(x)

            df['class'] = y_pred

            x.to_csv(f_name, header=False, index=False)

            print("\nAccuracy: {}".format(accuracy_score(y, y_pred)))
            print("Precision: {}".format(precision_score(
                y, y_pred, average='weighted')))
            print("Recall: {}\n".format(
                recall_score(y, y_pred, average='weighted')))

        else:
            break

    except(FileNotFoundError):
        print("ERROR: FILE NOT FOUND")
    except(NameError):
        print("ERROR: NO TRAINED MODEL FOUND")
