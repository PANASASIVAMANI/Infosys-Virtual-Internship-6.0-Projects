import pandas as pd
import os
from datasets import Dataset # Only import what's needed from datasets
# Removed: from datasets import load_metric (now in 'evaluate')
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
# Note: You will need to run the following command in Colab before running the script:
# !pip install evaluate

# Configuration
MODEL_NAME = "t5-small"
TRAINING_FILE = "nlp_data_train.csv"
OUTPUT_DIR = "finetuned_t5_summarizer"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 150
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5

# === NEW CONFIGURATION FOR DATASET COLUMNS ===
# Set this to 'summary' for Summarization task or 'paraphrase' for Paraphrasing task
TARGET_COLUMN = "summary" 
# =============================================

def load_and_preprocess_data():
    """
    Loads the training data, tokenizes it, and prepares it for the Trainer.
    """
    print("Loading data from CSV...")
    
    if not os.path.exists(TRAINING_FILE):
        print(f"Error: Training file '{TRAINING_FILE}' not found. Please ensure it is uploaded to Colab.")
        return None, None

    # Load T5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    
    # Load data with highly robust parsing to handle malformed lines and commas within fields
    try:
        # --- FIX: Explicitly set the engine to 'python' for more robust parsing of commas in text ---
        df = pd.read_csv(
            TRAINING_FILE,
            sep=',',
            encoding='utf-8',
            engine='python', # Use the Python engine for better handling of complicated CSV structures (slower, but more robust)
            on_bad_lines='skip' 
        )
        # -------------------------------------------------------------
        
    except Exception as e:
        print(f"Critical Error during CSV loading: {e}")
        print("Please check if your file is truly a standard comma-separated CSV.")
        return None, None

    # Ensure text and target columns are present and handle NaNs by dropping those rows
    required_cols = ['text', TARGET_COLUMN]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in the CSV. Found columns: {df.columns.tolist()}")
            return None, None
            
    df.dropna(subset=required_cols, inplace=True)
    
    # Convert Pandas DataFrame to Hugging Face Dataset
    datasets = Dataset.from_pandas(df)
    
    # Split the dataset into train and test (90% train, 10% test)
    datasets = datasets.train_test_split(test_size=0.1, seed=42)
    
    print(f"Total training examples: {len(datasets['train'])}")
    print(f"Total test examples: {len(datasets['test'])}")
    print(f"Task configured: Input column 'text' -> Target column '{TARGET_COLUMN}'")
    
    # Tokenizing function
    def preprocessing_function(examples):
        # Determine the prefix based on the task (optional but good practice)
        prefix = "summarize: " if TARGET_COLUMN == "summary" else "paraphrase: "
        
        # Tokenize inputs (text column)
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True)

        # Tokenize targets (summary or paraphrase column)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[TARGET_COLUMN], max_length=MAX_TARGET_LENGTH, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("\nTokenizing and preprocessing training data...")
    # Map the preprocessing function to the entire dataset
    # We remove 'id' and 'reference_text' as they are not needed for training
    tokenized_datasets = datasets.map(
        preprocessing_function,
        batched=True,
        remove_columns=['id', 'reference_text', 'summary', 'paraphrase'], # <--- UPDATED COLUMNS
    )
    
    # Remove the target column that is NOT being used to avoid conflicts, but keep the current target and 'text' for logging/debugging if necessary.
    # The 'remove_columns' argument in the map function handles this efficiently.
    
    return tokenizer, tokenized_datasets


def train_model(tokenizer, tokenized_datasets):
    """
    Initializes the model and the Trainer, and starts the fine-tuning process.
    """
    print("Loading Model...")
    # Load the T5 model for Sequence-to-Sequence generation
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=LEARNING_RATE, # Use learning rate as weight decay
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True, # Enable mixed precision training for T4 GPU
        report_to="none", # Disable logging to external services
    )

    # Use a data collator to correctly pad the batches of data
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"], # Using the 10% test split as validation
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("\n--- Starting Fine-Tuning ---")
    
    # Start training
    trainer.train()
    
    print("\n--- Training Complete! Saving model... ---")
    
    # Save the final model and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✅ Fine-tuned model saved successfully to the '{OUTPUT_DIR}' directory.")
    print("The next step is to download this folder and integrate it into your backend.py.")


if __name__ == "__main__":
    # In a new Colab cell, run the following first to ensure all dependencies are met:
    # !pip install transformers datasets pandas torch accelerate evaluate
    
    # Load, preprocess, and tokenize the data
    tokenizer, datasets = load_and_preprocess_data()

    if tokenizer and datasets:
        train_model(tokenizer, datasets)