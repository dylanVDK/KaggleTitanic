#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel

def status(feature):
    print('Step -->     ', feature, ': 100%')


def getFullData():
    # reading data
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')

    # Remove column 'Survived' before merging
    # merging data
    fullData = pd.concat([train, test]).reset_index(drop=True)
    # Remove useless data column
    fullData.drop(['PassengerId', 'Ticket'], inplace=True, axis=1)
    print(fullData.tail())
    status('Prepare data')
    return fullData


def addTitles(fullData):
    # global fullData

    # 1     =    Officer;
    # 2     =    royal / speciaL(jonkheer, dr);
    # 3     =    Femme civil;
    # 4     =    Homme civil
    TitleList = {"Capt": 1, "Col": 1, "Major": 1, "Rev": 1,
                 "Jonkheer": 2, "Don": 2, "Dona": 2, "Sir": 2, "Dr": 2, "the Countess": 2, "Lady": 2,
                 "Mme": 3, "Mlle": 3, "Ms": 3, "Mrs": 3, "Miss": 3,
                 "Mr": 4, "Master": 4,
                 }
    # extract the title from each name
    fullData['Title'] = fullData['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    # Handle all titles with the title list / categorie
    fullData['Title'] = fullData.Title.map(TitleList)

    # Remove useless data column
    fullData.drop(['Name'], inplace=True, axis=1)

    status('Title added')
    return fullData


def addFamilyStatus(fullData):
    # introducing a new feature : the size of families (including the passenger)
    fullData['FamilySize'] = fullData['Parch'] + fullData['SibSp'] + 1

    # introducing other features based on the family size
    fullData['FamilyStatus'] = fullData['FamilySize'].map(lambda s: 1 if s >= 2 else 0)

    # Remove useless data column
    fullData.drop(['FamilySize'], inplace=True, axis=1)
    status('Family Status added')
    return fullData


def addAge(row, listAgeMedian):
    condition = (
            (listAgeMedian['Sex'] == row['Sex']) &
            (listAgeMedian['Title'] == row['Title']) &
            (listAgeMedian['Pclass'] == row['Pclass']) &
            (listAgeMedian['FamilyStatus'] == row['FamilyStatus'])
    )
    return listAgeMedian[condition]['Age'].values[0]


def cleanAge(fullData):
    listAgeMedian = fullData.groupby(['Sex', 'Title', 'Pclass', 'FamilyStatus']).median().reset_index()[['Sex', 'Title', 'Pclass', 'FamilyStatus', 'Age']]

    # call addAge() and replace missing value in row 'Age' with average/mean median
    fullData['Age'] = fullData.apply(lambda row: addAge(row, listAgeMedian) if np.isnan(row['Age']) else row['Age'], axis=1)
    status('Cleaned age')
    return fullData


def cleanFare(fullData):
    # replace missing value by the mean fare.
    fullData.Fare.fillna(fullData.Fare.mean(), inplace=True)
    status('Cleaned fare')
    return fullData

def cleanEmbarked(fullData):
    # global fullData
    # get data without missing value
    tmpData = fullData.dropna(subset=['Embarked'])

    # find the most frequent value (it's S)
    frequentEmbarked = tmpData['Embarked'].value_counts().idxmax()

    # Replace the missing value by the most frequent value in row
    fullData.Embarked.fillna(frequentEmbarked, inplace=True)

    # create new row for each embarked
    embarked_dummies = pd.get_dummies(fullData['Embarked'], prefix='Embarked')
    fullData = pd.concat([fullData, embarked_dummies], axis=1)
    fullData.drop('Embarked', axis=1, inplace=True)
    status('Cleaned embarked')
    return fullData


def cleanSex(fullData):
    # convert male to 1 and female to 0
    fullData['Sex'] = fullData['Sex'].map({'male': 1, 'female': 0})
    status('Cleaned sex')
    return fullData


def cleanCabin(fullData):
    # global fullData
    # replace missing number cabins with X
    fullData.Cabin.fillna('X', inplace=True)

    # kepp only the cabin letter
    fullData['Cabin'] = fullData['Cabin'].map(lambda c: c[0])

    # create new row gor each letter cabin
    cabin_dummies = pd.get_dummies(fullData['Cabin'], prefix='Cabin')
    fullData = pd.concat([fullData, cabin_dummies], axis=1)

    # Remove useless data column
    fullData.drop('Cabin', axis=1, inplace=True)
    status('Cleaned cabin')
    return fullData


# transform all attribute in float via pd.getDummies
def minmaxScaler(fullData):
    std = MinMaxScaler()
    X = std.fit_transform(fullData)
    data_std = pd.DataFrame(X, columns=fullData.columns)
    return data_std


def trainData(data_std):
    data_train = data_std.iloc[0:891, :]
    dataTrain = data_train.drop('Survived', axis=1)
    return dataTrain

def trainDataSurvived(data_std):
    data_train = data_std.iloc[0:891, :]
    dataSurvived = data_train['Survived']
    return dataSurvived

def testData(data_std):
    data_test = data_std.iloc[891:1309, :]
    data_test = data_test.drop('Survived', axis=1)
    return data_test

def testDataSurvived(data_std):
    data_train = data_std.iloc[891:1309, :]
    dataSurvived = data_train['Survived']
    return dataSurvived

def randomForestTree(dataTrain, dataSurvived):
    tree = RandomForestClassifier(n_estimators=20, max_features='sqrt', bootstrap=True)

    tree.fit(dataTrain, dataSurvived)

    tree_score = cross_val_score(tree, dataTrain, dataSurvived, scoring="neg_mean_squared_error", cv=10)
    tree_rmse = np.sqrt(-tree_score)

    print("\n\nNeg Mean Sgraphique comparatif binaire pythonquared Error\n", tree_rmse)
    print("\n\nRandom Forest Classifier\nMoyenne", tree_rmse.mean())
    print("Ecart-type", tree_rmse.std())
    return tree

def getPredict(tree, dataTest):
    predict = tree.predict(dataTest)
    return predict

def renderSurvived(predict):
    test = pd.read_csv('./test.csv')
    output = np.column_stack((test['PassengerId'], predict))
    res = pd.DataFrame(output.astype('int'), columns=['PassengerId', 'Survived'])
    res.to_csv('./gender_submission.csv', index=False)
    print("\n\nPrediction Head \n", res.head())
    print("\n\nPrediction Tail \n", res.tail())


if __name__ == '__main__':
    # Prepare Data
    fullData = getFullData()
    fullData = addTitles(fullData)
    fullData = addFamilyStatus(fullData)
    fullData = cleanAge(fullData)
    fullData = cleanFare(fullData)
    fullData = cleanEmbarked(fullData)
    fullData = cleanSex(fullData)
    fullData = cleanCabin(fullData)
    print(fullData.head())

    # Lissage des donnees [MinMax]
    data_std = minmaxScaler(fullData.astype('float'))

    # Separation des donnees en 3 datasets

    # Donnees entrainee [Part 1]
    dataTrain = trainData(data_std)
    # Donnees entrainee label [Part 1]
    dataSurvived = trainDataSurvived(data_std)
    # Donnees non entrainee
    dataTest = testData(data_std)
    # Donnees entrainee label [Part 1]
    dataTestSurvived = testDataSurvived(data_std)
    # Algorithme
    tree = randomForestTree(dataTrain, dataSurvived)
    # affutage des labels
    model = SelectFromModel(tree, prefit=True)
    dataTrainReduce = model.transform(dataTrain)
    dataTestReduce = model.transform(dataTest)

    tree = tree.fit(dataTrainReduce, dataSurvived)
    predict = getPredict(tree, dataTestReduce)

    renderSurvived(predict)