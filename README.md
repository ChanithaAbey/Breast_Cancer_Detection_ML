Breast Cancer Detection using Machine Learning

This project uses a machine learning model based on Logistic Regression to classify tumors as either malignant (cancerous) or benign (not cancerous) based on the analysis of several diagnostic features. This is achieved through the use of the Breast Cancer Wisconsin dataset available through scikit-learn, developed using Python along with relevant data science libraries.

## Features
- Uses scikit-learn's built-in breast cancer dataset (569 samples, 30 features)
- Logistic Regression classifier with convergence handling
- Accuracy evaluation on test data
- Command-line interface for custom sample prediction
- Well-commented, modular Python code

---

## Technologies Used
- Python 3.11
- scikit-learn
- pandas
- warnings
- random

---

## How to Run the Project

1. Clone this repository:
```bash
git clone https://github.com/chanithaa/Breast_Cancer_Detection_ML.git
cd Breast_Cancer_Detection_ML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the script:
```bash
python breast_cancer_detection.py
```

4. Enter the number of test samples to predict when prompted.

---

## Example Output

```
First 5 samples of tumor features (each column = 1 patient, each row = 1 feature):

mean radius         17.99     20.57 ...
mean texture        10.38     17.77 ...
...

Model accuracy: 0.9649

Sample Data: [ ...features... ]
Predicted Class: 0, Actual Class: 1
```

---

## License
This project is open-source and available under the MIT License.

---

## Author
**Chanitha Abeygunawardena**  
[GitHub Profile](https://github.com/chanithaa)

