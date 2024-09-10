import pandas as pd


def categorize_tumor_size(df,threshold):
    df['size'] = df['SHAPE_Volume(mL)_tum'].apply(lambda x: 0 if x <= threshold else 1)
    return df

def read_data(path, num_labels=2, location=None, size=None,threshold = 1.0):
    # Load the data
    df = pd.read_csv(path, header=0)
    df = categorize_tumor_size(df,threshold)

    # Print initial size and column names
    print("Initial DataFrame size:", len(df))
   

    # Filter by location if specified
    if location:
        if location in df.columns:
            df = df[df[location] == 1]
            print(f"DataFrame size after location filter '{location}':", len(df))
        else:
            print(f"Location column '{location}' does not exist in the DataFrame.")

    # Filter by size if specified
    if size is not None:
        if 'size' in df.columns:
            df = df[df['size'] == size]
            print(f"DataFrame size after size filter '{size}':", len(df))
        else:
            print("Size column does not exist in the DataFrame.")

    # Add label column if it doesn't exist
    if 'label' not in df.columns:
        if num_labels == 3:
            df["label"] = df["tumor_evolution"].apply(lambda x: 0 if x <= -30 else (1 if x >= 20 else 2))
        elif num_labels == 2:
            df["label"] = df["tumor_evolution"].apply(lambda x: 0 if x <= -30 else 1)

    # Print size before and after dropping NA values
    print("DataFrame size before dropping NA:", len(df))
    df = df.dropna()
    print("DataFrame size after dropping NA:", len(df))

    # Group by patient_id
    grouped_data = df.groupby(['patient_id'])

    return grouped_data, df




