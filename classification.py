import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

'''
import and clean train set
'''
df = pd.read_csv('data/train.csv')
df['Age'] = df['Age'].fillna(df['Age'].mean())
df = df.drop(
    ['Ticket', 'Cabin', 'Name'], 
    axis=1
)
df = df.set_index('PassengerId')

'''
define X, y and split the train set
'''
X = df.drop(
    'Survived', 
    axis = 1
)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2, 
                                                    random_state = 42)

'''
define transformers for both categorical and numerical values
'''
cat_pipe = make_pipeline(SimpleImputer(strategy = 'most_frequent'),
                         OneHotEncoder(handle_unknown = 'ignore', sparse = False))
num_pipe = make_pipeline(SimpleImputer(strategy='mean'), 
                         MinMaxScaler())
feature_transform = ColumnTransformer(transformers=[('num', num_pipe, ['Age', 'Fare']),
                                                    ('cat', cat_pipe, ['Pclass', 'Embarked', 'Parch', 'Sex']), 
                                                    ('do_nothing', 'passthrough', ['SibSp'])])

'''
transform and fit train set, transform test set
'''
X_train_trans = feature_transform.fit_transform(X_train)
X_test_trans = feature_transform.transform(X_test)

'''
define model, fit it and calculate predition and probability
'''
m = LogisticRegression()
m.fit(X_train_trans, y_train)
proba = m.predict_proba(X_test_trans)
y_pred = m.predict(X_test_trans)

'''
print scores
'''
print(m.score(X_train_trans, y_train), 
      m.score(X_test_trans, y_test))

print(accuracy_score(y_pred, y_test), 
      precision_score(y_pred, y_test), 
      recall_score(y_pred, y_test), 
      f1_score(y_pred, y_test))

'''
import, transform test set and calculate predition
'''
test = pd.read_csv(
    'data/test.csv', 
    sep=','
)
test_trans = feature_transform.transform(test)
predition = m.predict(test_trans)

'''
define output df with predition for every passengerId and save it as .csv file
'''
output = pd.DataFrame(
    {'PassengerId':test['PassengerId'], 
     'Survived':predition}
)
output.to_csv(
    'data/submission.csv', 
    index=False
)