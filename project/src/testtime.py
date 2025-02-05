import pandas as pd

# Load the dataset
file_path = "..data/ESK10705.csv"  # Replace with the actual file path
df = pd.read_csv(file_path, encoding="utf-8", encoding_errors="ignore")

# Reset index if necessary
df_reset = df.reset_index()

# Convert the datetime column to standard 24-hour format
df_reset['index'] = pd.to_datetime(df_reset['index'], format='%Y-%m-%d %I:%M:%S %p', errors='coerce')

# Rename the column
df_reset.rename(columns={'index': 'Date Time Hour Beginning'}, inplace=True)

# Remove duplicate columns if necessary
df_reset = df_reset.loc[:, ~df_reset.columns.duplicated()]

# Display the cleaned dataset
print(df_reset.head())
