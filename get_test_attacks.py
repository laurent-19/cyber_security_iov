import pandas as pd

# Read the DataFrame from data/test_data
df = pd.read_csv('data/test_data.csv')

# Filter the DataFrame to get the attack labeled entries
attack_df = df[df['label'] == 'ATTACK']

# Output the attack labeled entries to another file
attack_df.to_csv('data/attack_labeled_entries.csv', index=False)