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

COMBINATIONS = [
    ("base", "class_weight", "base-model"),
    ("dropout", "class_weight", "dropout-model"),
    ("regularization", "class_weight", "regularization-model"),
    ("complete", "class_weight", "complete-model"),
    ("advanced", "class_weight", "advanced-model"),
    ("base", "smote", "smote-base"),
    ("dropout", "smote", "smote-dropout"),
    ("regularization", "smote", "smote-regularization"),
    ("complete", "smote", "smote-complete"),
    ("advanced", "smote", "smote-advanced"),
]
