### Description

This implementation of Naive Bayes Classification is a supervised learning algorithm that uses Bayes theorem to classify data into different categories. It is based on the assumption that the features of the data are independent of each other. The algorithm works by calculating the prior probabilities of the classes, the likelihoods of the features given the class labels, and the prior probabilities of the features. It then uses these values to calculate the posterior probability of the class labels for new data. The algorithm can be used for both binary and multi-class classification problems.

### Instruction

#### Prerequisites

- Python (<=3.7)
- Numpy
- Pandas
- Scikit-learn (for performance metrics)
- python-docx

#### To Compile & Execute Code:

'''
python3 test.py
'''

#### To Run Program:

Menu contains 3 operations namely, 'Train Classifier', 'Test Classifier', 'Exit'. Train the classifier by running the program and selection option 1 and enter the file name of training data. Test the classifier by selecting opetion 2 and enter the file name of test data and output file name.
