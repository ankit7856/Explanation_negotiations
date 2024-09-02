import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# Load the mixed dataset
df_mixed_new = pd.read_csv('merged_shuffled_dataset.csv')

# Check the initial target audience distribution for verification
print("Initial target audience distribution:\n", df_mixed_new['target_audience'].value_counts())

# Correctly encode target labels using LabelEncoder
label_encoder = LabelEncoder()
df_mixed_new['target_label'] = label_encoder.fit_transform(df_mixed_new['target_audience'])

# Verify the label encoding
print("Label encoding classes:", label_encoder.classes_)  # This should output ['expert' 'layperson']
print("Encoded labels distribution:\n", df_mixed_new['target_label'].value_counts())

# Check the actual encodings to ensure correctness
encoded_expert = label_encoder.transform(['expert'])[0]
encoded_layperson = label_encoder.transform(['layperson'])[0]
print(f"Encoded 'expert' as: {encoded_expert}")  # Expected: 0
print(f"Encoded 'layperson' as: {encoded_layperson}")  # Expected: 1

# Define the custom dataset class
class ExplanationDataset(Dataset):
    def __init__(self, dataframe):
        self.labels = dataframe['target_label'].values
        self.texts = dataframe['explanation'].values
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding='max_length', max_length=512
        )
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Classification requires long tensor
        return item

# Function to train a model
def train_model(df, model_name):
    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(df))
    val_size = len(df) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        ExplanationDataset(df), [train_size, val_size]
    )

    # Load BERT model for classification
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=len(label_encoder.classes_)
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./results_{model_name}',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'./logs_{model_name}',
        logging_steps=10,
        evaluation_strategy="epoch",  
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,  
        metric_for_best_model="eval_loss", 
        logging_first_step=True
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Train the model
    trainer.train()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Save the model and tokenizer
    model.save_pretrained(f'bert-finetuned-{model_name}')
    tokenizer.save_pretrained(f'bert-finetuned-{model_name}')

# Train the model on the mixed dataset
train_model(df_mixed_new, 'Nego_BERT')
