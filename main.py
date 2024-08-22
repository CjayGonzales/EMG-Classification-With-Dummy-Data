import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Generate dummy EMG signal data
def generate_dummy_emg_data(samples=1000, features=8):
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(samples, features)  # Generate random EMG signal data
    y = np.random.randint(0, 5, size=samples)  # Random labels (1 to 5)
    return X, y

# Generate the data
X, y = generate_dummy_emg_data()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the XGBoost classifier
model = XGBClassifier(objective='multi:softmax', num_class=5, eval_metric='mlogloss', use_label_encoder=False)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Example prediction
print(f"Predicted labels: {y_pred[:10]}")
print(f"True labels: {y_test[:10]}")
