import joblib
from sklearn.datasets import load_iris

# Load the trained model
model = joblib.load("iris_model.pkl")

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Make predictions
predictions = model.predict(X)
print("Predictions:", predictions)