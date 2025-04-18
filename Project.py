#Personal Project 1 2025 Chanitha Abeygunawardena 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas
import random
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def model():
    # Load Dataset
    breast_cancer = load_breast_cancer()
    dataset = pandas.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
    
    # Displays rows and columns of samples in dataset
    print("\nFirst 5 samples of tumor features (each column = 1 patient, each row = 1 feature):\n")
    print(dataset.head().T)

    """Features (tumor characteristics) and labels (1 for malignant, 0 for benign) 
    Note: Malignant means that the tumor IS CANCEROUS and Benign means that the tumor is NOT CANCEROUS"""
    tumor_features = breast_cancer.data
    diagnosis_labels = breast_cancer.target

    # Split the data into training and testing sets
    training_features, testing_features, training_labels, testing_labels = train_test_split(
        tumor_features, diagnosis_labels, test_size=0.2, random_state=42)

    # Logistic Regression model
    model = LogisticRegression(max_iter=3000)  # Limits iterations for convergence

    # Trains model using the training data
    model.fit(training_features, training_labels)

    # Predictions based on the test data
    predicted_labels = model.predict(testing_features)

    # Check Model Accuracy
    accuracy = accuracy_score(testing_labels, predicted_labels)
    print(f"\nModel accuracy: {accuracy}")

    return model, testing_features, testing_labels

def predict_new_data(model, testing_features, testing_labels):
    # Get a random sample from the test set and predict its class
    index = random.randint(0, len(testing_features) - 1)
    new_sample_data = testing_features[index].reshape(1, -1)  # Reshape the sample to fit the model's input shape

    # True label for comparison
    true_label = testing_labels[index]

    # Predicted label
    predicted_label = model.predict(new_sample_data)
    print("")
    print (f"Sample Data: {new_sample_data[0]}")
    print("")
    print (f"Predicted Class: {predicted_label[0]}, Actual Class: {true_label}\n")

try:
    # Get user input for number of samples to predict
    num = int(input("Enter the number of samples to predict: "))
except ValueError:
    print("Please enter an integer next time")

else:
    if num <= 0:
        print("Please enter a positive integer next time")
        exit(0)
    else:
        model, testing_features, testing_labels = model()
        for _ in range(num):
            predict_new_data(model, testing_features, testing_labels)
            

