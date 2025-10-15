import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import random

# Setting a random seed ensures that the sampling and splitting are reproducible.
random.seed(42)

def generate_and_save_dataset(dataset_name="xsum", total_rows=20000, test_size=0.2, output_prefix="nlp_data"):
    """
    Loads a public NLP dataset (XSum by default), samples it to a specific size, 
    splits it into training and evaluation sets, and saves them as CSV files.

    This function renames and duplicates columns to fit the required structure: 
    'id', 'text', 'summary', 'paraphrase', and 'reference'.

    The single 'summary' column from XSum is used as the ground truth for all three target columns 
    ('reference', 'summary', 'paraphrase') because XSum is primarily a summarization dataset.

    Args:
        dataset_name (str): The name of the dataset to load (e.g., 'xsum').
        total_rows (int): The total number of rows for the final dataset.
        test_size (float): The proportion of the dataset to include in the test split (e.g., 0.2 for 20%).
        output_prefix (str): Prefix for the output CSV files.
    """
    print(f"Starting dataset generation from: {dataset_name}...")
    
    try:
        # Load the 'train' split of the dataset
        # We explicitly map the required columns to the original dataset columns
        dataset = load_dataset(dataset_name, split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Convert to pandas DataFrame
    df = dataset.to_pandas()
    
    # 1. Rename columns to match the required structure
    # 'document' -> 'text' (Input)
    # 'summary' -> 'reference' (General Ground Truth)
    df = df.rename(columns={'document': 'text', 'summary': 'reference_text'})
    
    # 2. Duplicate the reference text for 'summary' and 'paraphrase' columns
    # In a real multi-task scenario, these columns would contain distinct ground truths.
    # For XSum, we use the reference summary for both:
    df['summary'] = df['reference_text']
    df['paraphrase'] = df['reference_text']
    
    # 3. Select the requested columns and ensure the correct order
    df = df[['id', 'text', 'summary', 'paraphrase', 'reference_text']]

    print(f"Original dataset size: {len(df)} rows.")

    if len(df) < total_rows:
        print(f"Warning: Dataset size ({len(df)}) is less than requested size ({total_rows}). Using all available data.")
        sample_df = df
    else:
        # Sample the required number of rows randomly for a diverse subset
        sample_df = df.sample(n=total_rows, random_state=42).reset_index(drop=True)
    
    print(f"Sampled dataset size: {len(sample_df)} rows.")
    
    # Split the sampled data into training and evaluation (test) sets (80/20 split)
    train_df, test_df = train_test_split(
        sample_df, 
        test_size=test_size, 
        random_state=42,
        shuffle=True
    )

    # Save the resulting dataframes to CSV files
    train_file = f"{output_prefix}_train.csv"
    test_file = f"{output_prefix}_eval.csv" 
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print("\n" + "=" * 50)
    print("✅ Dataset generation complete!")
    print(f"Total Rows Generated: {len(sample_df)}")
    print(f"Train File Saved: {train_file} ({len(train_df)} rows)")
    print(f"Evaluation File Saved: {test_file} ({len(test_df)} rows)")
    print("Columns in generated CSVs: 'id', 'text', 'summary', 'paraphrase', 'reference_text'")
    print("=" * 50)


if __name__ == '__main__':
    # Run the function to generate the dataset, explicitly setting the total row count.
    generate_and_save_dataset(total_rows=20000)