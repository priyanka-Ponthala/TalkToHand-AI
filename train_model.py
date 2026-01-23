import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load Data
df = pd.read_csv('master_data.csv')

# 2. Separate "Features" (X) and "Labels" (y)
# X = all columns except the last one (the landmarks)
# y = the last column (the name of the sign)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 3. Convert words (HELLO, YES) into numbers (0, 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the encoder so we can translate numbers back to words later
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# 4. Split data: 80% for learning, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Build the Neural Network Model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(np.unique(y_encoded)), activation='softmax') # Output layer
])

# 6. Compile & Train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training started... please wait.")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 7. Save the final "Brain"
model.save('sign_model.h5')
print("Model saved as sign_model.h5")