# Breast Cancer Detection using Machine Learning

This project uses a machine learning model based on Logistic Regression to classify tumors as either malignant (cancerous) or benign (not cancerous) based on the analysis of several diagnostic features. This is achieved through the use of the Breast Cancer Wisconsin dataset available through scikit-learn, developed using Python along with relevant data science and machine learning libraries.

## Features
- Uses scikit-learn's built-in breast cancer dataset (569 samples, 30 features)
- Logistic Regression model with convergence handling
- Accuracy evaluation on test data
- Command-line interface for custom sample prediction
- Well-commented, modular Python code

---

## Technologies Used
- Python 3.11
- scikit-learn
- pandas
- random

---

## How to Run the Project

1. Clone this repository:
```bash
git clone https://github.com/ChanithaAbey/Breast_Cancer_Detection_ML
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
Enter the number of samples to predict: 5

First 5 samples of tumor features (each column = 1 patient, each row = 1 feature):

                                   0            1            2           3            4
mean radius                17.990000    20.570000    19.690000   11.420000    20.290000
mean texture               10.380000    17.770000    21.250000   20.380000    14.340000
mean perimeter            122.800000   132.900000   130.000000   77.580000   135.100000
mean area                1001.000000  1326.000000  1203.000000  386.100000  1297.000000
mean smoothness             0.118400     0.084740     0.109600    0.142500     0.100300
mean compactness            0.277600     0.078640     0.159900    0.283900     0.132800
mean concavity              0.300100     0.086900     0.197400    0.241400     0.198000
mean concave points         0.147100     0.070170     0.127900    0.105200     0.104300
mean symmetry               0.241900     0.181200     0.206900    0.259700     0.180900
mean fractal dimension      0.078710     0.056670     0.059990    0.097440     0.058830
radius error                1.095000     0.543500     0.745600    0.495600     0.757200
texture error               0.905300     0.733900     0.786900    1.156000     0.781300
perimeter error             8.589000     3.398000     4.585000    3.445000     5.438000
area error                153.400000    74.080000    94.030000   27.230000    94.440000
smoothness error            0.006399     0.005225     0.006150    0.009110     0.011490
compactness error           0.049040     0.013080     0.040060    0.074580     0.024610
concavity error             0.053730     0.018600     0.038320    0.056610     0.056880
concave points error        0.015870     0.013400     0.020580    0.018670     0.018850
symmetry error              0.030030     0.013890     0.022500    0.059630     0.017560
fractal dimension error     0.006193     0.003532     0.004571    0.009208     0.005115
worst radius               25.380000    24.990000    23.570000   14.910000    22.540000
worst texture              17.330000    23.410000    25.530000   26.500000    16.670000
worst perimeter           184.600000   158.800000   152.500000   98.870000   152.200000
worst area               2019.000000  1956.000000  1709.000000  567.700000  1575.000000
worst smoothness            0.162200     0.123800     0.144400    0.209800     0.137400
worst compactness           0.665600     0.186600     0.424500    0.866300     0.205000
worst concavity             0.711900     0.241600     0.450400    0.686900     0.400000
worst concave points        0.265400     0.186000     0.243000    0.257500     0.162500
worst symmetry              0.460100     0.275000     0.361300    0.663800     0.236400
worst fractal dimension     0.118900     0.089020     0.087580    0.173000     0.076780

Model accuracy: 0.956140350877193

Sample Data: [1.570e+01 2.031e+01 1.012e+02 7.666e+02 9.597e-02 8.799e-02 6.593e-02
 5.189e-02 1.618e-01 5.549e-02 3.699e-01 1.150e+00 2.406e+00 4.098e+01
 4.626e-03 2.263e-02 1.954e-02 9.767e-03 1.547e-02 2.430e-03 2.011e+01
 3.282e+01 1.293e+02 1.269e+03 1.414e-01 3.547e-01 2.902e-01 1.541e-01
 3.437e-01 8.631e-02]

Predicted Class: 0, Actual Class: 0


Sample Data: [1.016e+01 1.959e+01 6.473e+01 3.117e+02 1.003e-01 7.504e-02 5.025e-03
 1.116e-02 1.791e-01 6.331e-02 2.441e-01 2.090e+00 1.648e+00 1.680e+01
 1.291e-02 2.222e-02 4.174e-03 7.082e-03 2.572e-02 2.278e-03 1.065e+01
 2.288e+01 6.788e+01 3.473e+02 1.265e-01 1.200e-01 1.005e-02 2.232e-02
 2.262e-01 6.742e-02]

Predicted Class: 1, Actual Class: 1


Sample Data: [1.298e+01 1.935e+01 8.452e+01 5.140e+02 9.579e-02 1.125e-01 7.107e-02
 2.950e-02 1.761e-01 6.540e-02 2.684e-01 5.664e-01 2.465e+00 2.065e+01
 5.727e-03 3.255e-02 4.393e-02 9.811e-03 2.751e-02 4.572e-03 1.442e+01
 2.195e+01 9.921e+01 6.343e+02 1.288e-01 3.253e-01 3.439e-01 9.858e-02
 3.596e-01 9.166e-02]

Predicted Class: 1, Actual Class: 1


Sample Data: [2.060e+01 2.933e+01 1.401e+02 1.265e+03 1.178e-01 2.770e-01 3.514e-01
 1.520e-01 2.397e-01 7.016e-02 7.260e-01 1.595e+00 5.772e+00 8.622e+01
 6.522e-03 6.158e-02 7.117e-02 1.664e-02 2.324e-02 6.185e-03 2.574e+01
 3.942e+01 1.846e+02 1.821e+03 1.650e-01 8.681e-01 9.387e-01 2.650e-01
 4.087e-01 1.240e-01]

Predicted Class: 0, Actual Class: 0


Sample Data: [1.181e+01 1.739e+01 7.527e+01 4.289e+02 1.007e-01 5.562e-02 2.353e-02
 1.553e-02 1.718e-01 5.780e-02 1.859e-01 1.926e+00 1.011e+00 1.447e+01
 7.831e-03 8.776e-03 1.556e-02 6.240e-03 3.139e-02 1.988e-03 1.257e+01
 2.648e+01 7.957e+01 4.895e+02 1.356e-01 1.000e-01 8.803e-02 4.306e-02
 3.200e-01 6.576e-02]

Predicted Class: 1, Actual Class: 1

```

---


