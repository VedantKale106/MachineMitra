  # MachineMitra

This Flask application enables users to upload a CSV file, select features and a target variable, train a machine learning model, and make predictions. The app supports various regression models, such as Linear Regression, Random Forest, Gradient Boosting, and Decision Tree. Additionally, it provides functionality to export the trained model and encoders used for preprocessing.

## Features

- **Upload CSV**: Allows users to upload their dataset in CSV format.
- **Select Features and Target Variable**: Users can choose which columns from the dataset will be used as input features and the target variable.
- **Train Model**: Offers the ability to train different machine learning models, including:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Decision Tree Regressor
  - Support Vector Regressor (SVR)
  - K-Nearest Neighbors (KNN)
  - AdaBoost Regressor
  - Lasso Regression
  - Ridge Regression
  - Elastic Net
  - Extra Trees Regressor
  - Histogram-Based Gradient Boosting Regressor


- **Make Predictions**: Enables users to input example data and receive predictions based on the trained model.
- **Export Model and Encoders**: Users can download the trained model and preprocessing encoders as pickle files for later use or deployment.
- **User Manual**: A detailed guide on how to use the app is available.

## Application Flow

1. **Upload Data**: Start by uploading a CSV file containing your dataset.
2. **Select Features**: Choose the input features and the target variable for the model.
3. **Train the Model**: Select the desired model type and train it using the selected features and target.
4. **Make Predictions**: Input data examples to predict outcomes using the trained model.
5. **Export**: Download the trained model and preprocessing encoders as pickle files.

## Model Options

- **Linear Regression**: A basic model that assumes a linear relationship between input features and the target variable.
- **Random Forest Regressor**: An ensemble model that builds multiple decision trees and merges them to improve accuracy and control overfitting.
- **Gradient Boosting Regressor**: A powerful model that builds models in a sequential manner, each trying to correct the errors of the previous one.
- **Decision Tree Regressor**: A non-linear model that splits the data into subsets based on the feature that results in the most significant difference in the target variable.
- **Support Vector Regressor (SVR)**: A model that uses support vector machines to find a function that deviates minimally from the actual target values while keeping the complexity of the function low.
- **K-Nearest Neighbors (KNN)**: A model that predicts the target variable by looking at the 'k' closest data points in the feature space and taking an average of their target values.
- **AdaBoost Regressor**: An ensemble method that combines multiple weak models to create a strong model by adjusting the weights of observations based on the errors of previous models.
- **Lasso Regression**: A type of linear regression that includes a penalty term to reduce the magnitude of coefficients, effectively performing feature selection.
- **Ridge Regression**: A type of linear regression that includes a penalty term to shrink coefficients but doesn’t perform feature selection.
- **Elastic Net**: A combination of Lasso and Ridge regression that balances between feature selection and coefficient shrinkage.
- **Extra Trees Regressor**: An ensemble method similar to Random Forest but uses a more random approach to split points in the trees, which often leads to better performance.
- **Histogram-Based Gradient Boosting Regressor**: A variation of Gradient Boosting that uses histogram-based binning for faster and more memory-efficient training.

## Data Preprocessing

- **Categorical Encoding**: The app automatically encodes categorical variables using `LabelEncoder`.
- **Data Integrity Checks**: Ensures there are no missing values or incorrect data types before training.

## Export Functionality

- **Model Export**: The trained machine learning model can be exported as a `.pkl` file.
- **Encoders Export**: The encoders used for categorical data are also exportable as a `.pkl` file.

## User Manual

To access the user manual, navigate to the `/user_manual` endpoint, where you can find detailed instructions on how to use the application, including steps for uploading files, selecting features, training models, making predictions, and exporting data.

## About

For more information about the app, visit the `/about` endpoint.

