# Multivariate Linear Regression
<br>
Linear Regresion using multiple variable. Regression is a type of supervised learning algorithm which learns from labelled data or data with target variable specified for each training example.<br><br>
The model equation in case of univariate linear regression is:<br>
H(X) = θ0 + θ1\*X1 + θ2\*X2 + ... + θn\*Xn
<br>
Here H(X) is the hypothesis function, X is the input feature, θ0, θ1 ... θn are the model parameters. H(X) is a linear function and hence it is called linear regression.
<br><br>
This project takes data about a number of apartments like the area, number of rooms and price of the apartment.<br><br>
The purpose of this project is to learn model parameters from the training data and hence be able to predict apartment price given the area and number of rooms.<br><br>

The gradient descent algorithm is an iterative process, it converges to optimal model parameters after multiple iterations. At each iteration it moves closer towards the optimal parameters. Gradient descent algorithm selects those model parameters which give the minimum error/cost values.<br><br>
On running main.m file the following values of model parameters are calculated:<br><br>
&nbsp;θ0 : 334302.063993<br>
&nbsp;θ1 : 68386.331827<br>
&nbsp;θ2 : 68386.331827<br><br>
The plot for cost vs number of iterations is as follows, You can see that the cost decreases with the number of iterations:<br>
<img src="https://github.com/kailashmaurya/Machine-Learning/blob/master/Multivariate%20Linear%20Regression/costVSiterations.jpg" style="margin-right: auto; margin-left: auto;"><br>

When we use the trained model to predict values we get the following results:<br><br>
&nbsp;Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): $288828.6829<br><br>

This is how multivariate linear regression learns from the training data using gradient descent algorithm and normal equations.