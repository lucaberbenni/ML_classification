This project aims to predict the survival of passengers on the Titanic using machine learning techniques. It utilizes the popular Titanic dataset, which contains information about passengers such as age, gender, ticket class, fare, and more.

### Data Preprocessing

The project begins with importing and cleaning the training dataset. Missing values in the 'Age' column are filled with the mean age, and irrelevant columns such as 'Ticket', 'Cabin', and 'Name' are dropped. The 'PassengerId' column is set as the index for easy reference.

### Model Training

The dataset is split into training and testing sets, with 80% of the data used for training. A pipeline is created to handle both categorical and numerical features. Categorical features are imputed with the most frequent value and then one-hot encoded, while numerical features are imputed with the mean and then scaled using min-max scaling. The `LogisticRegression` algorithm is chosen as the predictive model.

The training set is transformed and fitted using the pipeline, and the test set is transformed accordingly. The model is then trained on the transformed training set, and predictions and probabilities are calculated for the transformed test set.

### Model Evaluation

Several evaluation metrics are computed to assess the performance of the model. These include accuracy, precision, recall, and F1 score. The scores are printed to provide insights into the model's accuracy and predictive capabilities.

### Making Predictions

An additional test dataset is imported for making predictions on unseen data. The test dataset is transformed using the same pipeline and then used to predict survival using the trained model. The predictions are stored in an output dataframe along with the corresponding 'PassengerId' and saved as a .csv file for further analysis.

This project demonstrates the application of logistic regression for survival prediction on the Titanic dataset. It serves as a starting point for understanding and exploring more advanced machine learning techniques for classification tasks.

Feel free to explore the code and dataset provided to gain insights into the Titanic dataset and experiment with different models and techniques for improving prediction accuracy.

*Note: The dataset used in this project is for educational purposes only and does not represent real-world data.*
