### Description

This implementation of Naive Bayes Classification is a supervised learning algorithm that uses Bayes theorem to classify data into different categories. It is based on the assumption that the features of the data are independent of each other. The algorithm works by calculating the prior probabilities of the classes, the likelihoods of the features given the class labels, and the prior probabilities of the features. It then uses these values to calculate the posterior probability of the class labels for new data. The algorithm can be used for both binary and multi-class classification problems.

Additionally 2 Cross Validation methods were used to build better model. Those are Simple Random K-Fold Cross Validation and Stratified K-Fold Cross Validation.

### Instruction

#### Prerequisites

- Python (<=3.7)
- numpy
- pandas
- python-docx

#### To Compile & Execute Code:

'''
python3 test.py
'''

#### To Run Program:

Menu contains 4 operations namely, 'Train', 'Classify', 'Test accuracy', 'K-Fold CV', 'Stratified K-Fold CV' and 'Exit'. Train the classifier by running the program, selecting option 1 and enter the file name of metadata and training data. Predict inputs by selecting option 2 and enter the file name and output of file name. Test the classifier by selecting option 3 and enter the file name of test data. Build better classifier using K-Fold CV by selecting option 4 and enter the file name of metadata, training data and number of folds. Another approach to build better classifier with stratified sampling is by using Stratified K-Fold CV by selecting option 5 and enter the file name of metadata, training data and number of folds.
