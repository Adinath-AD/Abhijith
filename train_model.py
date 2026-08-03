import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ============================================================
# 1. GENERATE SYNTHETIC SMART GRID DATASET
# ============================================================

np.random.seed(42)

samples = 5000

data = []

for i in range(samples):

    # --------------------------------------------------------
    # Randomly choose NORMAL or FAULT
    # --------------------------------------------------------

    fault = np.random.choice([0, 1], p=[0.7, 0.3])

    if fault == 0:

        # NORMAL CONDITION
        voltage = np.random.uniform(210, 250)
        current = np.random.uniform(0.5, 7)
        temperature = np.random.uniform(20, 50)
        humidity = np.random.uniform(30, 70)

    else:

        # FAULT CONDITION
        fault_type = np.random.choice([
            "undervoltage",
            "overvoltage",
            "overcurrent",
            "overtemperature",
            "combined"
        ])

        if fault_type == "undervoltage":

            voltage = np.random.uniform(150, 190)
            current = np.random.uniform(1, 7)
            temperature = np.random.uniform(20, 50)
            humidity = np.random.uniform(30, 70)

        elif fault_type == "overvoltage":

            voltage = np.random.uniform(260, 280)
            current = np.random.uniform(1, 7)
            temperature = np.random.uniform(20, 50)
            humidity = np.random.uniform(30, 70)

        elif fault_type == "overcurrent":

            voltage = np.random.uniform(210, 250)
            current = np.random.uniform(8, 10)
            temperature = np.random.uniform(20, 60)
            humidity = np.random.uniform(30, 70)

        elif fault_type == "overtemperature":

            voltage = np.random.uniform(210, 250)
            current = np.random.uniform(1, 7)
            temperature = np.random.uniform(70, 100)
            humidity = np.random.uniform(30, 70)

        else:

            # COMBINED FAULT

            voltage = np.random.uniform(160, 200)
            current = np.random.uniform(8, 10)
            temperature = np.random.uniform(70, 100)
            humidity = np.random.uniform(30, 70)

    # --------------------------------------------------------
    # Calculate power
    # --------------------------------------------------------

    power = voltage * current

    data.append([
        voltage,
        current,
        temperature,
        humidity,
        power,
        fault
    ])


# ============================================================
# 2. CREATE DATAFRAME
# ============================================================

columns = [
    "voltage",
    "current",
    "temperature",
    "humidity",
    "power",
    "fault"
]

df = pd.DataFrame(data, columns=columns)


# ============================================================
# 3. SAVE DATASET
# ============================================================

df.to_csv("dataset.csv", index=False)

print()
print("==========================================")
print(" DATASET CREATED")
print("==========================================")

print("Number of samples:", len(df))

print()
print(df.head())

print()
print("Dataset saved as: dataset.csv")


# ============================================================
# 4. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

X = df[
    [
        "voltage",
        "current",
        "temperature",
        "humidity",
        "power"
    ]
]

y = df["fault"]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)


print()
print("Training Random Forest...")

model.fit(X_train, y_train)


# ============================================================
# 7. TEST MODEL
# ============================================================

y_prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_prediction
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print()
print("==========================================")
print(" RANDOM FOREST RESULTS")
print("==========================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        y_prediction
    )
)


# ============================================================
# 9. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    "model.pkl"
)

print()
print("==========================================")
print(" MODEL SAVED")
print("==========================================")

print("model.pkl created successfully.")


# ============================================================
# 10. TEST SOME SAMPLE VALUES
# ============================================================

print()
print("==========================================")
print(" SAMPLE PREDICTIONS")
print("==========================================")


test_samples = pd.DataFrame([
    {
        "voltage": 230,
        "current": 3.5,
        "temperature": 30,
        "humidity": 55,
        "power": 230 * 3.5
    },

    {
        "voltage": 170,
        "current": 4,
        "temperature": 32,
        "humidity": 50,
        "power": 170 * 4
    },

    {
        "voltage": 230,
        "current": 9,
        "temperature": 35,
        "humidity": 55,
        "power": 230 * 9
    },

    {
        "voltage": 230,
        "current": 5,
        "temperature": 80,
        "humidity": 50,
        "power": 230 * 5
    }
])


predictions = model.predict(test_samples)

probabilities = model.predict_proba(test_samples)


for i in range(len(test_samples)):

    if predictions[i] == 0:
        status = "NORMAL"
    else:
        status = "FAULT"

    confidence = max(probabilities[i]) * 100

    print()
    print(
        "Sample",
        i + 1,
        "→",
        status,
        "| Confidence:",
        round(confidence, 2),
        "%"
    )