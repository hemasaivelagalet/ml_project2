Student Performance Prediction Model 🎓📊
Objective

The goal of this project is to develop a machine learning model that predicts students' academic performance (e.g., final grades) based on various factors such as attendance, study habits, and socio-economic background.
Dataset

The dataset used in this project is the Student Performance Dataset from the UCI Machine Learning Repository. It contains data on students' academic and demographic factors.

    Dataset Link: UCI Student Performance Dataset
    Key Features:
        Demographic: Gender, age, parental education, family size.
        Behavioral: Attendance, study time, free time, extracurricular activities.
        Institutional: Past grades, school support.
    Target Variable: G3 (Final Grade).

Project Workflow

    Data Loading:
        Load the dataset from a .csv file.
        Preview the structure and clean the data.

    Preprocessing:
        Encode categorical variables using one-hot encoding.
        Split data into training and testing sets.

    Exploratory Data Analysis (EDA):
        Analyze correlations between features and the final grade.
        Visualize the grade distribution.

    Model Training:
        Train a Random Forest Regressor to predict final grades.

    Model Evaluation:
        Evaluate the model using Mean Squared Error (MSE) and R2 Score.

    Feature Importance:
        Analyze which features contribute the most to the prediction.

Installation

To run this project, follow the steps below:

    Clone this repository or download the project files.
    Install the required Python libraries:

    pip install pandas numpy matplotlib seaborn scikit-learn

    Download the Student Performance Dataset and save it as student-mat.csv in the project directory.

Usage

    Run the script:

    python student_performance.py

    Outputs:
        Correlation heatmap of features.
        Distribution of final grades.
        Model evaluation metrics (MSE, R2).
        Feature importance plot.

Results

    Model Performance:
        Mean Squared Error: Displayed during execution
        R2 Score: Displayed during execution
