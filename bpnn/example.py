import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import standardize_train, standardize_test
from model import BPRegressor
import matplotlib.pyplot as plt

df = pd.read_excel('/Users/micheal/Documents/Datasets/fetch_california_housing.xlsx')

X = df.drop('target', axis=1).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_std, mean, std = standardize_train(X_train)
X_test_std = standardize_test(X_test, mean, std)

y_train_std, y_mean, y_std = standardize_train(y_train.reshape(-1,1))
y_train_std = y_train_std.flatten()

model = BPRegressor(layers=[X_train_std.shape[1], 32, 8, 1], lr=0.3, epochs=1000)
model.fit(X_train_std, y_train_std)

y_pred = model.predict(X_test_std) * y_std + y_mean

print("Test MSE:", mean_squared_error(y_test, y_pred))
print("Test R2:", r2_score(y_test, y_pred))

plt.plot(model.losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()
