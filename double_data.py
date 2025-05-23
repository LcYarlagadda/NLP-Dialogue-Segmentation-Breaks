import pandas as pd

# Load the original dataset from a CSV file
file_path = 'data/train.csv'  # Adjust the path as necessary
data = pd.read_csv(file_path)

# Double the dataset by appending it to itself
doubled_data = pd.concat([data, data], ignore_index=True)

# Save the doubled dataset back to a CSV file
doubled_data_path = 'data/doubled_train.csv'  # Adjust the path as necessary
doubled_data.to_csv(doubled_data_path, index=False)
