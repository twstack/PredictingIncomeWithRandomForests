def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Step 1: Load the data
income_data = pd.read_csv("income.csv", delimiter=", ", header=0)

# Step 2: Print the first row of the data
print(income_data.iloc[0])

# Step 3: Separate the labels from the data
labels = income_data[["income"]]

# Step 4: Change 'sex' and 'country' columns to numerical values
income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)
income_data["country-int"] = income_data["native-country"].apply(lambda row: 0 if row == "United-States" else 1)

# Step 5: Select columns for prediction
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "country-int"]]

# Step 6: Create the Random Forest model
forest = RandomForestClassifier(random_state=1)

# Step 7: Split the data, Train the data, Test the accuracy of the Random Forest model
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)
forest.fit(train_data, train_labels)
print(forest.score(test_data, test_labels))