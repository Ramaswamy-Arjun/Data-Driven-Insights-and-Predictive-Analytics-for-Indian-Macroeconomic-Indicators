import pandas as pd
import glob

# Define the folder path containing the transposed files
folder_path = r"C:\Users\arjun\Ensemble Learning"  # Update this path if necessary
file_pattern = "transposed_gst_data-*.xlsx"  # Match files with this pattern (for years 2017-2018 to 2023-2024)

# Use glob to find all matching files
file_paths = glob.glob(f"{folder_path}\\{file_pattern}")

# List to hold data from each file
data_frames = []

# Load each file, add a 'Year' column, and append it to the list
for file_path in file_paths:
    # Extract the year range from the file name (e.g., '2017-2018')
    year_range = file_path.split("-")[-1].replace(".xlsx", "")
    
    # Load the data
    df = pd.read_excel(file_path)
    
    # Add the 'Year Range' column
    df['Year Range'] = year_range
    
    # Append the DataFrame to the list
    data_frames.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(data_frames, ignore_index=True)

# Reorder to place 'Ladakh' at column index 34 if it exists
if 'Ladakh' in combined_df.columns:
    # Remove 'Ladakh' from its current position
    columns = list(combined_df.columns)
    columns.remove('Ladakh')
    
    # Insert 'Ladakh' at index 34
    columns.insert(34, 'Ladakh')
    
    # Reorder the DataFrame
    combined_df = combined_df[columns]

# Save the combined DataFrame to a new file
combined_df.to_excel(f"{folder_path}\\combined_gst_data_2017_2024.xlsx", index=False)
# Optionally save as CSV
# combined_df.to_csv(f"{folder_path}\\combined_gst_data_2017_
