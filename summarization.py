import pandas as pd
import json
import matplotlib.pyplot as plt
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import datasets
from datasets import Dataset, DatasetDict
import evaluate

# Load and preprocess dataset
df = pd.read_json('ca_test_data_final_OFFICIAL.jsonl', lines=True)
selected_columns = ['text_len', 'bill_id', 'sum_len']
df = df.drop(columns=selected_columns)

# Function to remove punctuation
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

nltk.download('stopwords')
stop_words = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
df['text'] = df['text'].apply(remove_punctuations)

# Removing patterns from text
pattern = r'(SECTION \d+\s?)|\([^)]*\)|(SEC. \d+\s?)'
df['text'] = df['text'].str.replace(pattern, '', regex=True)

# Splitting dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
tds = Dataset.from_pandas(train_df)
vds = Dataset.from_pandas(test_df)
ds = DatasetDict({'train': tds, 'test': vds})

# Tokenizer and model initialization
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Preprocess function
prefix = "summarize: "
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    summary_list = examples['summary']
    labels = tokenizer(text_target=summary_list, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

ds = ds.map(preprocess_function, batched=True)

# Data collator and metrics
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

# Training
training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_billsum_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,  # Consider changing to True if you want to push to Hugging Face Hub
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save model and tokenizer
tokenizer.save_pretrained('Summerization_model_directory')
model.save_pretrained('Summerization_model_directory')



# Inference
# text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."


# from transformers import pipeline

# summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
# summarizer(text)


# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
# inputs = tokenizer(text, return_tensors="pt").input_ids

