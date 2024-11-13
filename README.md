# Diabetes Prediction using Artificial Neural Networks

This project utilises artificial neural networks (ANNs) to predict the likelihood of diabetes in individuals based on health-related data. By analysing key health metrics, the model aims to assist in early detection and risk assessment of diabetes, contributing to proactive healthcare and informed decision-making.

## Project Overview

Diabetes is a growing global health concern, and early prediction is essential for effective management and prevention. This project leverages deep learning, specifically artificial neural networks, to analyze patient data and predict diabetes risk, enabling healthcare providers and individuals to take timely action.

## Dataset

The model is trained on Pima Indians Diabetes dataset from Kaggle containing various health metrics for individuals, including:
- Glucose level
- Blood pressure
- Body Mass Index (BMI)
- Insulin level
- Skin thickness
- Age
- Other relevant features

These features are commonly associated with diabetes risk, allowing the model to learn patterns and make predictions effectively.

## Features

- **Diabetes Prediction**: Predicts the likelihood of diabetes based on input features.
- **Artificial Neural Network**: A deep learning model with multiple hidden layers to capture complex relationships within the data.
- **Model Evaluation**: Uses accuracy, precision, recall, and F1-score to assess model performance.

## Model Architecture

The artificial neural network model is composed of multiple fully connected layers, with ReLU activation functions in hidden layers and a sigmoid activation function in the output layer to predict the probability of diabetes.

### Model Layers
- **Input Layer**: Takes the input features (e.g., glucose, BMI).
- **Hidden Layers**: Multiple hidden layers with ReLU activation for non-linearity.
- **Output Layer**: A single neuron with a sigmoid activation function to output the probability of diabetes.

## Usage

1. **Prepare the Data**: Preprocess the dataset by normalizing the input features for optimal model performance.
2. **Train the Model**: Run the training script to train the neural network model on the dataset.
3. **Make Predictions**: Use the trained model to make predictions on new data.  

## [Click here to view the project in detail](https://github.com/Naveen-Baburaj/Diabetes-Prediction-Using-Artificial-Neural-Networks/blob/main/Diabetes%20(1).ipynb)  



## Results

The neural network model achieved promising results, with high accuracy and balanced precision and recall, making it suitable for early-stage diabetes prediction. The model demonstrates the potential of deep learning in medical data analysis and early health risk detection.


## Technologies Used

- **Python**: Core programming language.
- **TensorFlow/Keras**: Used for building and training the artificial neural network.
- **NumPy**: For efficient numerical operations.
- **Pandas**: For data manipulation and preprocessing.
- **Matplotlib & Seaborn**: For data visualisation.

---

This project showcases the potential of artificial neural networks in predictive healthcare, offering a data-driven approach to diabetes risk assessment. By harnessing machine learning, it provides a foundation for developing intelligent, preventive healthcare solutions.
