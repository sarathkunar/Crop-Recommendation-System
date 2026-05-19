import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("crop_recommendation.csv")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Encode crop labels
crop_encoder = LabelEncoder()

df["label"] = crop_encoder.fit_transform(df["label"])

# Features
X = df.drop("label", axis=1)

# Target
y = df["label"]

# Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Save files
pickle.dump(model, open("crop_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(crop_encoder, open("crop_encoder.pkl", "wb"))

print("Model Saved Successfully")