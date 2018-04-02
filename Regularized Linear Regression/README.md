# Linear Regression Dataset:
<br>
The UCI Wine Quality dataset lists 11 chemical measurements of 4898 white wine samples as well as an overall quality per sample, as determined by wine connoisseurs.
See winequality-white.csv, it is split into training, validation and test sets randomly in the preprocessing code. The purpose of this project is to train the linear regression model and hence use linear regression to predict wine quality from the chemical measurement features.
<br><br>
# Methods Description:
<ol>
<li>data_processing</li>
This method reads the dataset from the file winequality-white.csv and splits it into training, validation and test sets. The X sets holds the input features and the Y sets hold the output.<br><br>
<li>linear_regression_noreg</li>
Takes the training dataset to learn the model parameters.<br><br>
<li>regularized_linear_regression</li>
To address the challenge such as overfitting, we add regularization. I have considered L2 regularization with the linear regression model.<br>
<img src="https://github.com/kailashmaurya/Machine-Learning/blob/master/Regularized%20Linear%20Regression/regularized.JPG" style="margin-right: auto; margin-left: auto;"><br>
where lambda > 0 is a hyper-parameter used to control the complexity of the resulting model. When lambda = 0, the model reduces to the usual (unregularized) linear regression. 
For lambda > 0 the objective function balances between two terms: (1 )the data-dependent quadratic loss function and (2) a function of the model parameters(hence model complexity).<br><br>
<li>tune_lambda</li>
Trains the linear regression model using different values of lambda using thetraining dataset and then calculates the test error using the validation dataset. Hence it decides the best lambda based on which one gives the least validation error.<br><br>
<li>test_error</li>
Calculates the error by compairing the difference between the actual result and the predicted values.<br>
</ol><br>

# Output Observed:
<br>
The output observed by running the file linear_regression.py is as follows:<br><br>

dimensionality of the model parameter is 12.<br>
model parameter is  [  2.166e+02   1.145e-01  -1.824e+00  -1.065e-02   1.037e-01   1.546e-01   3.416e-03   2.347e-04  -2.173e+02   8.348e-01   7.366e-01   1.153e-01]<br>
<br><br>
dimensionality of the model parameter is 12<br>
lambda = 5.0, model parameter is [  6.324e-01  -2.854e-02  -1.641e+00  -3.577e-02   2.803e-02  -1.362e-01   5.259e-03  -8.421e-04   5.888e-01   2.887e-01   4.162e-01   3.768e-01]<br>
<br><br>
tuning lambda, the best lambda =  0.001
<br>

Mean Square Error(MSE) on test is 0.512

# Runner:
<br>
Make sure linear_regression.py and the dataset csv file are in the same directory. Also, use python3 to run the program.
