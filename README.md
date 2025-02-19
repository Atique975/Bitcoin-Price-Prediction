# Bitcoin Price Prediction Using Support Vector Regression (SVR)
## 📌 Overview
This project implements Bitcoin price prediction using a Support Vector Regression (SVR) model. The dataset consists of historical Bitcoin prices, and the goal is to predict future prices for the next 30 days.

## 🚀 Features
Data Preprocessing: Cleans and prepares the dataset for modeling.
Feature Engineering: Creates a prediction column shifted by 30 days.
Machine Learning Model: Uses Support Vector Regression (SVR) to predict Bitcoin prices.
Model Evaluation: Splits data into training and testing sets, calculates accuracy, and compares predictions with actual values.
## 📂 Dataset
Columns:

Date (removed during preprocessing)
Price (Bitcoin closing price)
Example Data:
| Price | Prediction |
|--------|-------------|
| 45000 | 46000 |
| 45500 | 46200 |

## 📊 Model Workflow
### Preprocessing:
Drops the Date column.
Creates a shifted "Prediction" column by 30 days.
### Feature Selection:
Converts data into NumPy arrays for training/testing.
### Train/Test Split:
Uses an 80/20 split for model evaluation.
### Model Training:
Implements Support Vector Regression (SVR) with an RBF kernel.
### Predictions:
Evaluates accuracy on test data.
Forecasts Bitcoin prices for the next 30 days.
## 📌 Code Overview
### 🏗️ Data Preprocessing
df.drop(['Date'], axis=1, inplace=True)  # Remove Date column

df['Prediction'] = df[['Price']].shift(-30)  # Shift price column up by 30 days

### 📈 Feature Selection
x = np.array(df.drop(['Prediction'], axis=1))[:-30]  # Independent variable

y = np.array(df['Prediction'])[:-30]  # Dependent variable

### 🏋️ Train/Test Split
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

### 🤖 Model Training & Prediction
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)

svr_rbf.fit(xtrain, ytrain)

## Predict next 30 days
svm_prediction = svr_rbf.predict(predictionDays_array)

print(svm_prediction)

## 📉 Results
SVR Accuracy: Prints model performance score.

Next 30-Day Predictions: [7817.49887569 7864.68961127 8489.43837767 9408.78777776 9488.03727658
 8140.16667578 8628.76394866 7920.95659363 7568.36233686 7815.44781066
 7768.71930057 7539.26034891 8085.44338188 8450.58546821 8197.48527099
 8427.16738259 8299.0397825  8780.1897048  8128.0162555  8092.27188563
 8076.33420951 8409.57589865 8623.77743776 8173.27477694 8080.69714849
 8078.9966232  8668.19771953 7465.73224643 7665.35637587 7655.90537629]


