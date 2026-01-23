import pandas as pd
import glob
import os

# 1. Find all CSV files in the folder
csv_files = glob.glob("*.csv")

# 2. Combine them into one big list
combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# 3. Save the master dataset
combined_df.to_csv("master_data.csv", index=False)

print(f"Combined {len(csv_files)} files into master_data.csv")
print(f"Total frames collected: {len(combined_df)}")