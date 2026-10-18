import keras
import pandas as pd
from keras import models
from keras import layers

# Read the data file
data = pd.read_csv('Life Expectancy Data.csv')

# Rename some columns. The original names are appallingly inconsistent with respect to capitalization and
# extra spaces and the beginning, at the end, and even in the middle of column names.
data = data.rename(columns={'Life expectancy ':'Life Expectancy',
                            'infant deaths':'Infant Deaths',
                            'percentage expenditure':'Percentage Expenditure',
                            'Measles ':'Measles',
                            ' BMI ':'BMI',
                            'under-five deaths ':'Under-five Deaths',
                            'Total expenditure':'Total Expenditure',
                            'Diphtheria ':'Diphtheria',
                            ' HIV/AIDS':'HIV/AIDS',
                            ' thinness  1-19 years':'Thinness 1-19 Years',
                            ' thinness 5-9 years':'Thinness 5-9 Years',
                            'Income composition of resources':'Income Composition of Resources'
                            })

# Keep only the columns we're interested in
data = data[['Life Expectancy', 'Alcohol', 'Hepatitis B', 'Polio', 'Diphtheria', 'GDP', 'Schooling']]

# Discard any rows containing missing values
data.dropna(how='any', inplace=True)

# Shuffle the data for good measure
data = data.sample(frac=1)  # See https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows

# Scale the outputs
data['Life Expectancy'] /= 100

# Extract the correct answers
labels = data.pop('Life Expectancy')

#data = data.to_numpy()


# TODO
# Extract training, validation, and test sets. These should comprise 60%, 20%, and 20% of the data, respectively.
# You'll be defining variables X_train, y_train, X_valid, y_valid, X_test, and y_test.
# sixPct = 0.6*data.shape[0]
# twtPct = 0.2*data.shape[0]
#X_train = data[:, :sixPct-1,]
#y_train = data[:, :twtPct-1,]
# print(data)

#X_train = data.sample(frac=0.6)
#y_train = data.sample(frac=0.6)

n = len(data)
train_end = n * 6 // 10
valid_end = n * 8 // 10

X_train = data[:train_end]
y_train = labels[:train_end]
#print(X_train.shape)

X_valid = data[train_end:valid_end]
y_valid = labels[train_end:valid_end]
#print(X_valid.shape)

X_test = data[valid_end:]
y_test = labels[valid_end:]
#print(X_test.shape)

#print(X_train)

# Normalize all three sets, but based on the mean and standard deviation of the training data only to avoid snooping.
mean = data[:train_end].mean(axis=0)    
std = data[:train_end].std(axis=0)
X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std
X_test = (X_test - mean) / std

#print(X_train)
# Define the network
model = keras.Sequential(
        [
            layers.Input(shape=(6,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
# Compile the network
model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )
# Train the network
model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), verbose=2)

# ONLY AFTER YOU HAVE MADE ALL OF YOUR CHANGES, uncomment this section to test your network
# Test the network
print('\nTesting trained model:')
model.evaluate(X_test, y_test, verbose=2)