Student Performance Prediction Model

Project Overview

The Student Performance Prediction Model is a machine learning project aimed at predicting students' academic performance (e.g., final grades) based on various features such as attendance, study habits, and socio-economic factors. This project utilizes the Student Performance Dataset from the UCI Machine Learning Repository and implements a Random Forest Regressor to make predictions.

Key Features

Predict students' final grades based on input features.

Analyze the importance of factors like attendance, study time, and family background.

Perform exploratory data analysis (EDA) to understand feature correlations and grade distributions.

Evaluate the model's performance using metrics like Mean Squared Error (MSE) and R2 Score.

Visualize feature importance for better interpretability.

Dataset

Source: UCI Machine Learning Repository - Student Performance Dataset

Attributes:

Demographics: Gender, age, parent's job, family size.

Academic Factors: Past grades, study time, failures, school support.

Behavioral Factors: Absences, free time, extracurricular activities.

Target Variable: G3 (final grade).

Project Workflow

1. Data Preprocessing

Load the dataset.

Encode categorical variables using one-hot encoding.

Split data into training and testing sets.

2. Exploratory Data Analysis (EDA)

Correlation heatmap to identify relationships between features and the target variable.

Visualize the distribution of final grades.

3. Model Training

Use Random Forest Regressor as the predictive model.

Train the model using the training dataset.

4. Model Evaluation

Evaluate the model using:

Mean Squared Error (MSE)

R2 Score

Analyze the model's feature importance.

Installation

Prerequisites

Ensure you have Python installed along with the following libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

Install Dependencies

pip install pandas numpy matplotlib seaborn scikit-learn

Download the Dataset

Download the student-mat.csv file from the dataset link and save it in the same directory as the project script.

Running the Project

Save the project script as student_performance.py.

Run the script:

python student_performance.py

Outputs

Correlation Heatmap: Visualizes relationships between features and final grades.

Grade Distribution Plot: Displays the distribution of students' grades.

Model Metrics:

Mean Squared Error (MSE)

R2 Score

Feature Importance Plot: Highlights which factors contribute most to grade predictions.
