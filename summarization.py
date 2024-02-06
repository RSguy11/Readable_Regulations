import pandas as pd
import json 
import matplotlib.pyplot as plt
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import torch


df = pd.read_json('ca_test_data_final_OFFICIAL.jsonl', lines=True)

selected_columns = ['summary', 'title', 'sum_len']
dataf_Summaries = df[selected_columns].copy()

selected_columns[1] = 'bill_id'
df = df.drop(columns=selected_columns)

# Function to remove punctuation
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


stop = nltk.download('stopwords')

stop_words = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
# Apply the function to a DataFrame column
df['text]'] = df['text'].apply(remove_punctuations)

"""
The following code has been updated below to do the same thing more efficiently. It was changed to run in one line instead of making multiple calls to the dataframe.
# Removing the text between perentheses
pattern = r'\([^)]*\)'
# Replacing the matched text with an empty string
df['text'] = df['text'].str.replace(pattern, '', regex=True)
"""
# Removing the Section headings. Included under is the pattern to remvoe all the section words and the numbers after. It also removes all the text between perentheses and removes all the SEC. 
pattern = r'(SECTION \d+\s?)|\([^)]*\)|(SEC. \d+\s?)'
# Pattern = r'(?i)SECTION \d+\s?'

# Replacing the matched text with an empty string
df['text'] = df['text'].str.replace(pattern, '', regex=True)
textCell = df.iat[3, df.columns.get_loc('text')]
summaryCell = dataf_Summaries.iat[3, dataf_Summaries.columns.get_loc('summary')]

# print(textCell)
# print("\nSummary")
# print(summaryCell)



df_text = df['text']
df_summ = dataf_Summaries['summary']

# Split into train and test
train_text, test_text = train_test_split(df_text, test_size=0.2, random_state=42)
train_sum, test_sum = train_test_split(df_summ, test_size=0.2, random_state=42)


# Getting rid of "The people State California enact follows". Not entirely sure if that is neccesary.
train_text = train_text.str.replace(r'^.*?\:', '', regex=True)
test_text = train_text.str.replace(r'^.*?\:', '', regex=True)

from transformers import AutoTokenizer
# Load Tokenizer
checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# Define the prefix and preprocessing function
prefix = "summarize: "

def preprocess_function(text, summary):
    inputs = [prefix + doc for doc in text]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    # Convert the Pandas Series 'summary' to a list
    summary_list = summary.tolist() if isinstance(summary, pd.Series) else summary

    labels = tokenizer(text_target=summary_list, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



# Apply preprocessing function to training and testing sets
tokenized_train = preprocess_function(train_text, train_sum)
tokenized_test = preprocess_function(test_text, test_sum)


from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)



import evaluate

rouge = evaluate.load("rouge")


import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)


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
    fp16=True,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()



# Example of how to use the trained model for summarization
input_text = "The California State Legislature passed a bill to address climate change and promote sustainable energy practices. The bill outlines various measures to reduce carbon emissions, increase reliance on renewable energy sources, and encourage environmentally friendly practices across different sectors. It emphasizes the importance of transitioning to a low-carbon economy to mitigate the impacts of climate change on the state and its residents. The legislation also includes provisions for incentivizing green technology adoption and fostering innovation in the renewable energy sector."


inputs = tokenizer("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the generated summary
print("Generated Summary:", generated_summary)



# Inference
# text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."


# from transformers import pipeline

# summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
# summarizer(text)


# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
# inputs = tokenizer(text, return_tensors="pt").input_ids

