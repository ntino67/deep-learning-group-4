SOURCE = "Diabetes_012"

TARGET = "Diabetes"

POSITIVE_VALUE = 2.0

BINARY_COLS = [
    "Diabetes",
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]

INT_COLS = ["GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]

OUTLIER_COLS = ["BMI"]
