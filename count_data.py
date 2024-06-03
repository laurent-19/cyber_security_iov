import pandas as pd
import matplotlib.pyplot as plt

# Load the train, validation, and test datasets
train_data = pd.read_csv('data/train_data.csv')
val_data = pd.read_csv('data/val_data.csv')
test_data = pd.read_csv('data/test_data.csv')

# Function to count occurrences
def count_occurrences(df):
    label_counts = df['label'].value_counts().to_dict()
    category_counts = df['category'].value_counts().to_dict()
    specific_class_counts = df['specific_class'].value_counts().to_dict()
    return label_counts, category_counts, specific_class_counts

# Count occurrences in each dataset
train_label_counts, train_category_counts, train_specific_class_counts = count_occurrences(train_data)
val_label_counts, val_category_counts, val_specific_class_counts = count_occurrences(val_data)
test_label_counts, test_category_counts, test_specific_class_counts = count_occurrences(test_data)

# Create a summary DataFrame for combined data
combined_data = pd.concat([train_data, val_data, test_data], ignore_index=True)
combined_label_counts, combined_category_counts, combined_specific_class_counts = count_occurrences(combined_data)

summary_df = pd.DataFrame([
    {'Label': 'Benign', 'Category': '-', 'Class': '-', 'Count': combined_label_counts.get('BENIGN', 0)},
    {'Label': 'Attack', 'Category': 'DoS', 'Class': '-', 'Count': combined_category_counts.get('DoS', 0)},
    {'Label': 'Attack', 'Category': 'Spoofing', 'Class': 'GAS', 'Count': combined_specific_class_counts.get('GAS', 0)},
    {'Label': 'Attack', 'Category': 'Spoofing', 'Class': 'Steering Wheel', 'Count': combined_specific_class_counts.get('STEERING_WHEEL', 0)},
    {'Label': 'Attack', 'Category': 'Spoofing', 'Class': 'Speed', 'Count': combined_specific_class_counts.get('SPEED', 0)},
    {'Label': 'Attack', 'Category': 'Spoofing', 'Class': 'RPM', 'Count': combined_specific_class_counts.get('RPM', 0)},
])

print(summary_df)

# Plotting function
def plot_distribution(label_counts, category_counts, specific_class_counts, dataset_name):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.bar(label_counts.keys(), label_counts.values(), color='b')
    plt.title(f'{dataset_name} - Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    
    plt.subplot(3, 1, 2)
    plt.bar(category_counts.keys(), category_counts.values(), color='g')
    plt.title(f'{dataset_name} - Category Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    
    plt.subplot(3, 1, 3)
    plt.bar(specific_class_counts.keys(), specific_class_counts.values(), color='r')
    plt.title(f'{dataset_name} - Specific Class Distribution')
    plt.xlabel('Specific Class')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.show()

# Plot distributions for each dataset
plot_distribution(train_label_counts, train_category_counts, train_specific_class_counts, 'Train Data')
plot_distribution(val_label_counts, val_category_counts, val_specific_class_counts, 'Validation Data')
plot_distribution(test_label_counts, test_category_counts, test_specific_class_counts, 'Test Data')

