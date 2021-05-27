import pandas
from sklearn.linear_model import LinearRegression

dataset = pandas.read_csv("salary.csv")

x = dataset["Year Experience"].values.reshape(17,1)

y = dataset["Salary"]

model = LinearRegression()

model.fit( x , y )
y_hat = model.predict([[ 4.4 ]])
print("Salary for 4.4 Year Experince : ", y_hat) 

