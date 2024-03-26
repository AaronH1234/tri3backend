import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the titanic dataset
titanic_data = sns.load_dataset('titanic')

class TitanicRegression:
    # initializing attributes within a class.
    def __init__(self):
        self.dt = None
        self.logreg = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = None
    def initTitanic(td):
        td = titanic_data
        td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True) # dropping columns named 'alive', 'who', 'adult_male', 'class', 'embark_town', and 'deck' from the DataFrame
        td.dropna(inplace=True) # drop rows with at least one missing value, after dropping unuseful columns
        td['sex'] = td['sex'].apply(lambda x: 1 if x == 'male' else 0)
        td['alone'] = td['alone'].apply(lambda x: 1 if x == True else 0)

        # Encode categorical variables
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(td[['embarked']])
        onehot = enc.transform(td[['embarked']]).toarray()
        cols = ['embarked_' + val for val in enc.categories_[0]]
        td[cols] = pd.DataFrame(onehot)
        td.drop(['embarked'], axis=1, inplace=True)
        td.dropna(inplace=True)
    
    def runDecisionTree(td):
        # Build distinct data frames
        X = td.drop('survived', axis=1) # all except 'survived'
        y = td['survived'] # only 'survived'
        
        # X_train is the DataFrame for the training set.
        # X_test is the DataFrame for the test set.
        # y-train is the 'survived' status in the training set
        # y_test is the 'survived' status in the test set
        # 70% training, 30% testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train a decision tree classifier
        dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced')
        dt.fit(X_train, y_train)

        # Test the model
        y_pred = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('DecisionTreeClassifier Accuracy: {:.2%}'.format(accuracy))
        
    def runLogisticRegression(td):
        X = td.drop('survived', axis=1) # all except 'survived'
        y = td['survived'] # only 'survived'
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # Train a logistic regression model
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)

        # Test the model
        y_pred = logreg.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('LogisticRegression Accuracy: {:.2%}'.format(accuracy)) 
    
def initTitanic():
    global titanic_regression
    titanic_regression = TitanicRegression()
    titanic_regression.initTitanic()
    titanic_regression.runDecisionTree()
    titanic_regression.runLogisticRegression()
    
def predictSurvival(passenger):
    new_passenger = passenger.copy()
    
    # Preprocess the new passenger data
    new_passenger['sex'] = new_passenger['sex'].apply(lambda x: 1 if x == 'male' else 0)
    new_passenger['alone'] = new_passenger['alone'].apply(lambda x: 1 if x == True else 0)

    # Encode 'embarked' variable
    onehot = enc.transform(new_passenger[['embarked']]).toarray()
    cols = ['embarked_' + val for val in enc.categories_[0]]
    new_passenger[cols] = pd.DataFrame(onehot, index=new_passenger.index)
    new_passenger.drop(['name'], axis=1, inplace=True)
    new_passenger.drop(['embarked'], axis=1, inplace=True)

    # Predict the survival probability for the new passenger
    dead_proba, alive_proba = np.squeeze(logreg.predict_proba(new_passenger))

    # Print the survival probability
    print('Death probability: {:.2%}'.format(dead_proba))  
    print('Survival probability: {:.2%}'.format(alive_proba))
    result = {
        "death": 'Death probability: {:.2%}'.format(dead_proba), 
        "survival": 'Survival probability: {:.2%}'.format(alive_proba)
    }
    return result

    
if __name__ == "__main__":
    # Initialize the Titanic model
    initTitanic()

    # Predict the survival of a passenger
    passenger = {
        'name': ['Test'],
        'pclass': [2],
        'sex': ['male'],
        'age': [64],
        'sibsp': [1],
        'parch': [1],
        'fare': [16.00],
        'embarked': ['S'],
        'alone': [False]
    }
    print(predictSurvival(passenger))