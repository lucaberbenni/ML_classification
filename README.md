# ML_classification
machine learning model to predict passenger survival on Titanic.

1. **Data Preparation:**
   The code begins by importing the necessary libraries, including pandas for data handling. It reads the training dataset from a CSV file, performs some data cleaning operations, and sets the index to 'PassengerId'. Irrelevant columns such as 'Ticket', 'Cabin', and 'Name' are dropped.

2. **Data Splitting:**
   The code defines the feature matrix X and the target variable y. It then splits the data into training and testing sets using the train_test_split function from scikit-learn. The test set size is set to 20% of the original data.

3. **Data Transformation:**
   Two pipelines are defined: one for categorical features and one for numerical features. The categorical pipeline handles missing values by using the most frequent strategy and applies one-hot encoding to categorical variables. The numerical pipeline imputes missing values using the mean strategy and scales the numerical features using the MinMaxScaler. These pipelines are combined into a ColumnTransformer, which applies the respective transformations to the specified columns.

4. **Model Training and Evaluation:**
   A logistic regression model is instantiated, fitted with the transformed training data, and used to make predictions on the transformed test data. The code calculates various evaluation metrics, such as accuracy, precision, recall, and F1 score, using the predicted labels and the actual labels from the test set. These metrics provide an assessment of the model's performance.

5. **Prediction on Test Set:**
   The code imports the test dataset, applies the same data transformation pipeline used for the training set, and makes predictions on the transformed test data using the trained logistic regression model.

6. **Output Generation:**
   The predicted survival outcomes are stored in a DataFrame along with the corresponding passenger IDs. This DataFrame is saved as a CSV file named "submission.csv" in the "data" directory.
