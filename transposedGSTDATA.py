import pandas as pd

# Function to transpose and restructure GST data
def transpose_gst_data(file_path):
    # Load the data (only first sheet, if the sheet has the date format as headers)
    df = pd.read_excel(file_path, sheet_name=0, header=4)  # Starting from row 7

    # Drop unnecessary columns (any unnamed columns that aren't useful)
    df_cleaned = df.dropna(how='all', axis=1)  # Drop entirely empty columns

    # Transpose the data
    transposed_df = df_cleaned.set_index('State').T.reset_index()

    # Rename columns
    transposed_df.rename(columns={'index': 'Date'}, inplace=True)

    # Print the transposed structure for verification
    print(transposed_df.head())

    # Return the transposed DataFrame
    return transposed_df

# Example usage
file_path = 'Tax-Collection-on-GST-Portal-2018-2019-updated.xlsx'  # Update with your file path
transposed_df = transpose_gst_data(file_path)

# Check the transposed data
print(transposed_df.head())

# Optionally, save the transposed data to a new Excel file
transposed_df.to_excel('transposed_gst_data-2018-2019.xlsx', index=False)
