import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_paths = [
    'data/decimal_benign.csv',
    'data/decimal_DoS.csv',
    'data/decimal_spoofing-GAS.csv',
    'data/decimal_spoofing-RPM.csv',
    'data/decimal_spoofing-SPEED.csv',
    'data/decimal_spoofing-STEERING_WHEEL.csv'
]

# Split each dataframe into training, validation, and test sets
train_dataframes = []
val_dataframes = []
test_dataframes = []

for file in file_paths:
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42, shuffle=True)  # 0.25 * 0.8 = 0.2
    train_dataframes.append(train_df)
    val_dataframes.append(val_df)
    test_dataframes.append(test_df)

# Combine the training, validation, and test DataFrames separately
train_data = pd.concat(train_dataframes, ignore_index=True)
val_data = pd.concat(val_dataframes, ignore_index=True)
test_data = pd.concat(test_dataframes, ignore_index=True)

# Shuffle the combined dataframes
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
val_data = val_data.sample(frac=1, random_state=42).reset_index(drop=True)
test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the splits
train_data.to_csv('data/train_data.csv', index=False)
val_data.to_csv('data/val_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

