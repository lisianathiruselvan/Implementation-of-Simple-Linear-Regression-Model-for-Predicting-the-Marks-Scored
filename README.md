# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

Step 1: Use the standard libraries such as numpy, pandas, matplotlib.pyplot in python for the simple linear regression model for predicting the marks scored.

Step 2: Set variables for assigning dataset values and implement the .iloc module for slicing the values of the variables X and y.

Step 3: Import the following modules for linear regression; from sklearn.model_selection import train_test_split and also from sklearn.linear_model import LinearRegression.

Step 4: Assign the points for representing the points required for the fitting of the straight line in the graph.

Step 5: Predict the regression of the straight line for marks by using the representation of the graph.

Step 6: Compare the graphs (Training set, Testing set) and hence we obtained the simple linear regression model for predicting the marks scored using the given datas.

Step 7: End the program.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: T.LISIANA
RegisterNumber:  212222240053
*/
import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset)

# assigning hours to X & Scores to Y
X=dataset.iloc[:,:1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show

plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse

```

## Output:
![image](https://github.com/lisianathiruselvan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389971/f255707a-e539-4e03-ba8d-b67fd29e00c1)

![image](https://github.com/lisianathiruselvan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389971/dbc04532-6dac-49b2-b6a0-2b5787ed549a)

![image](https://github.com/lisianathiruselvan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389971/1440c817-e245-4068-baf1-30f955d36229)

![image](https://github.com/lisianathiruselvan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389971/a243c703-7bc2-4002-abc3-c1cb5370b3ed)

![image](https://github.com/lisianathiruselvan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389971/9175245e-fe42-4538-8349-98b8dfc7ce57)

![image](https://github.com/lisianathiruselvan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389971/4aa337cc-149c-4f21-bb22-9fb2031071ac)

![image](https://github.com/lisianathiruselvan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119389971/a98418fc-ae64-4af8-aeab-ac597057ccfb)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
