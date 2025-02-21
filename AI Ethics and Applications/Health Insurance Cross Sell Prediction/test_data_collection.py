import pandas as pd

# Assuming your dataset is loaded into a DataFrame
df = pd.read_clipboard('D:/Masters Projects/Master-In-AI/AI Ethics and Applications/Health Insurance Cross Sell Prediction/Data_Creation.xlsx')

#  Function to calculate unique values, count, and percentage
def get_unique_values(df):
    unique_info = []
    
    # Iterate over each column (feature) in the dataframe
    for column in df.columns:
        # Get value counts for each column
        value_counts = df[column].value_counts()
        
        # Add the counts and percentages for each unique value
        for value, count in value_counts.items():
            percentage = (count / len(df)) * 100
            unique_info.append([value, column, count, percentage])
    
    # Create a DataFrame to display the results
    unique_df = pd.DataFrame(unique_info, columns=['Unique Value', 'Feature Name', 'Count', 'Percentage'])
    return unique_df

# Get the unique values for each feature
result_df = get_unique_values(df)

# Save the result to a CSV file
result_df.to_csv('unique_values_summary.csv', index=False)

# Optionally, display the result
print(result_df)