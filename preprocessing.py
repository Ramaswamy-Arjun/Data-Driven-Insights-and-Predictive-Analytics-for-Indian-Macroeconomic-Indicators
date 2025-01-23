import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Define the file path
file_path = r"C:\Users\arjun\Ensemble Learning\combined_gst_data_2017_2024.xlsx"

# Load the data
df = pd.read_excel(file_path)

# Define the initial date and set increment
start_date = datetime.strptime("01-07-2017", "%d-%m-%Y")
date_increment = relativedelta(months=1)

# Update the 'Date' column in increments
dates = []
for i in range(0, len(df), 4):
    current_date = start_date + date_increment * (i // 4)
    dates.extend([current_date] * 4)

df['Date'] = dates

# Save the updated DataFrame to the same file
df.to_excel(file_path, index=False)

print("Dates updated and data saved successfully to the original file!")
