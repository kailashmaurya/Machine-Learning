# Univariate Linear Regression
<br>
Linear Regresion using one variable. Regression is a type of supervised learning algorithm which learns from labelled data or data with target variable specified for each training example.<br><br>
The model equation in case of univariate linear regression is:<br>
H(X) = θ-zero + θ-one\*X
<br>
Here H(X) is the hypothesis function, X is the input feature, θ-zero and θ-one are the model parameters. H(X) is a linear function and hence it is called linear regression.
<br><br>
This project takes data about population and profit earned(in 10,000s) about a number of cities.<br><br>
The initial plot of data is as follows:<br>
<img src="https://github.com/kailashmaurya/Machine-Learning/blob/master/Univariate%20Linear%20Regression/graphs/Data_Plot.JPG" style="margin-right: auto; margin-left: auto;"><br><br>
The gradient descent algorithm is an iterative process, it converges to optimal model parameters after multiple iterations. At each iteration it moves closer towards the optimal parameters. Gradient descent algorithm selects those model parameters which give the minimum error/cost values.<br><br>
On running main.m file the following values of model parameters are calculated:<br><br>
θ-zero : -3.630291<br>
θ-one  :  1.166362<br><br>
The model H(X) plotted with the input data is as follows:<br>
<img src="https://github.com/kailashmaurya/Machine-Learning/blob/master/Univariate%20Linear%20Regression/graphs/Model.jpg" style="margin-right: auto; margin-left: auto;"><br><br>
When we use the trained model to predict values we get the following results:<br><br>
&nbsp;For population = 35,000, we predict a profit of 4519.767868<br>
&nbsp;For population = 70,000, we predict a profit of 45342.450129<br><br>
Visualizing the cost function for possible values of θ-zero and θ-one we get the following graph:<br>
<img src="https://github.com/kailashmaurya/Machine-Learning/blob/master/Univariate%20Linear%20Regression/graphs/Cost_Function.jpg" style="margin-right: auto; margin-left: auto;"><br><br>
The error/cost is measured on the verticle axis and the model parameters on the other two.<br>
Plotting a contour diagram of the same cost function we get a contour disgram like this:
<img src="https://github.com/kailashmaurya/Machine-Learning/blob/master/Univariate%20Linear%20Regression/graphs/Cost_Contour.jpg" style="margin-right: auto; margin-left: auto;"><br><br>
The value of error/cost on all the points lying on same countour is same. The "X" mark represents the least error point on the diagram.
This is the point where the gradient descent algorithm converges and terminates.<br>
This is how univariate linear regression learns from the training data using gradient descent algorithm.
