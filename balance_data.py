import pandas as pd

# Load the dataset
df = pd.read_csv('data/train.csv')

# Separate majority and minority classes
majority_class = df[df['label'] == 0]
minority_class = df[df['label'] == 1]

# Get the number of instances in the minority class
minority_count = minority_class.shape[0]

# Get the number of instances in the majority class
majority_count = majority_class.shape[0]

# Calculate the ratio of minority to majority
ratio = majority_count // minority_count

# Duplicate instances in the minority class
balanced_minority = pd.concat([minority_class] * ratio, ignore_index=True)

# Concatenate majority class with balanced minority class
balanced_df = pd.concat([majority_class, balanced_minority])

# Shuffle the dataset to mix the instances
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

# Save or return the balanced dataset
balanced_df.to_csv('balanced_train.csv', index=False)
