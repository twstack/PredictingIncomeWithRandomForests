def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Step 2: Load the data
income_data = pd.read_csv("income.csv", delimiter = ", ", header = 0)

# Step 3: Print the first row of the data
print(income_data.iloc[0])

# Step 5: Separate the labels from the data
labels = income_data[["income"]]

# Step 6: Select columns for prediction
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week"]]

# Step 7: Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

# Step 8: Create the Random Forest model
forest = RandomForestClassifier(random_state = 1)

# Step 9: Fit the model
forest.fit(train_data, train_labels)

# Step 11: Test the model and print accuracy
print(forest.score(test_data, test_labels))

# Step 12: Change 'sex' column to numerical values
income_data["sex-int"] = income_data["sex"].apply(lambda row: 0 if row == "Male" else 1)

# Step 13: Add "sex-int" to data
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int"]]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)
forest.fit(train_data, train_labels)
print(forest.score(test_data, test_labels))

# Step 14 & 15: Look at the different values in "native-country" and map to numbers
print(income_data["native-country"].value_counts())
income_data["country-int"] = income_data["native-country"].apply(lambda row: 0 if row == "United-States" else 1)

# Step 16: Add "country-int" to data
data = income_data[["age", "capital-gain", "capital-loss", "hours-per-week", "sex-int", "country-int"]]
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)
forest.fit(train_data, train_labels)
print(forest.score(test_data, test_labels))