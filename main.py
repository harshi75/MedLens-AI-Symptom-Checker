import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")
X = data.drop("Disease", axis=1)
y = data["Disease"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
