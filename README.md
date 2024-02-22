# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
<br>2.Set variables for assigning dataset values.
<br>3.Import linear regression from sklearn.
<br>4.Assign the points for representing in the graph.
<br>5.Predict the regression for marks by using the representation of the graph.
<br>6.Compare the graphs and hence we obtained the linear regression for the given datas. 
  

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SUDHARSANA KUMAR S R
RegisterNumber: 212223240162
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
##Dataset:

![Screenshot 2024-02-22 091515](https://github.com/sudharsanakumar18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849110/4dfa8944-45ea-4520-be12-41ba501943e7)

##Head values:

![Screenshot 2024-02-22 091557](https://github.com/sudharsanakumar18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849110/c3b18e9c-6b34-42bd-8df9-294b279bfe66)

##Tail values:

![Screenshot 2024-02-22 091610](https://github.com/sudharsanakumar18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849110/861fc716-81ec-4975-96b4-db8ef2709178)

##X and Y values:

![Screenshot 2024-02-22 091654](https://github.com/sudharsanakumar18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849110/081f961c-e2c9-4574-98b6-a9018bc799aa)

##Training Set, Testing Set, MSE, MAE and RMSE:

![Screenshot 2024-02-22 090953](https://github.com/sudharsanakumar18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/138849110/e4523540-f46c-49d9-94eb-daff61e0084f)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
