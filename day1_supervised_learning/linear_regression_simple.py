from sklearn.linear_model import LinearRegression

dataset = {
    "sqft": [1200, 2400, 900],
    "price": [4800000, 9600000, 3600000]
}


x = [[value] for value in dataset['sqft']]
y = dataset['price']
model = LinearRegression()

model.fit(x, y)

predicted_price = model.predict([[1800]])

print(predicted_price)