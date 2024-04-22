# %%
def column_to_list(df, column_name):
    """
    Convert a column in a DataFrame to a list of lists.

    Parameters:
    - df: DataFrame
        The DataFrame containing the column to be converted.
    - column_name: str
        The name of the column to be converted to a list.

    Returns:
    - lists: list
        A list of lists where each inner list corresponds to a row in the specified column.
    """
    column_values = df[column_name].tolist()
    lists = [list(arr) for arr in column_values]
    return lists


# %%
import pandas as pd

# %%
id2label = {0: '0',
            1: 'B-PER', 
            2: 'I-PER',
            3: 'B-ORG',
            4: 'I-ORG',
            5: 'B-LOC',
            6: 'I-LOC'
           }

# %%
#Training data
trainin_data = pd.read_parquet('train-00000-of-00001.parquet')
training_labels_num = column_to_list(trainin_data, 'ner_tags')
training_labels = [[id2label[label_id] for label_id in sequence] for sequence in training_labels_num]
training_sent =  column_to_list(trainin_data, 'tokens')

#flatten to one list to be able to use myutils
train_flat_labels = sum(training_labels, [])
train_flat_sent = sum(training_sent, [])

# %%
#test data
test_data = pd.read_parquet('test-00000-of-00001.parquet')
test_labels_num = column_to_list(test_data, 'ner_tags')
test_labels = [[id2label[label_id] for label_id in sequence] for sequence in test_labels_num]

test_sent =  column_to_list(test_data, 'tokens')
test_index = [[i for i, _ in enumerate(sublist)] for sublist in test_labels]

#flatten to one list to be able to use myutils
test_flat_labels = sum(test_labels, [])
test_flat_sent = sum(test_sent, [])
test_flat_index = sum(test_index, [])

# %%
#validation data
validation_data = pd.read_parquet('validation-00000-of-00001.parquet')
dev_labels_num = column_to_list(validation_data, 'ner_tags')
dev_labels = [[id2label[label_id] for label_id in sequence] for sequence in dev_labels_num]

dev_sent =  column_to_list(validation_data, 'tokens')

#flatten to one list to be able to use myutils
dev_flat_labels = sum(dev_labels, [])
dev_flat_sent = sum(dev_sent, [])

# %%
#!pip install transformers

# %%
#!pip install ipywidgets

# %%
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# %%
inputs = []
for sentence in training_sent:
    inputt = tokenizer(sentence, is_split_into_words=True)
    inputs.append(inputt)

# %%
test_labels[0]


# %%
inputs[0].word_ids()

# %%
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

# %%
label2id = {label: id for id, label in id2label.items()}


# %%
train_flat_labels_num = [label2id[label_id] for label_id in train_flat_labels]


# %%


# %%
labels = training_labels
word_ids = []
for i in inputs:
    word_ids.append(i.word_ids())
word_ids_flat = sum(word_ids, [])
aligned_training = align_labels_with_tokens(train_flat_labels_num, word_ids_flat)


# %%
aligned_training

# %%
def list_to_sentences(lst):
    sentences = []
    current_sentence = []
    
    for item in lst:
        if item == -100:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
        else:
            current_sentence.append(item)
    
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences


sentences = list_to_sentences(aligned_training)
print(sentences)


# %%
to_zip_train = [[-100] + sublist + [-100] for sublist in sentences]

# %%
# Assuming inputs and to_zip_train are defined as provided

# Zip inputs with to_zip_train and add 'label' key to each item in inputs
for input_item, label_item in zip(inputs, to_zip_train):
    input_item['labels'] = label_item

# Print the updated inputs list
print(inputs[0])


# %%
inputs_dev = []
for sentence in dev_sent:
    inputt = tokenizer(sentence, is_split_into_words=True)
    inputs_dev.append(inputt)

# %%
dev_flat_labels_num = [label2id[label_id] for label_id in dev_flat_labels]


# %%
dev_flat_labels_num

# %%
word_ids = []
for i in inputs_dev:
    word_ids.append(i.word_ids())
word_ids_flat = sum(word_ids, [])
aligned_dev = align_labels_with_tokens(dev_flat_labels_num, word_ids_flat)

# %%
to_zip_dev = [[-100] + sublist + [-100] for sublist in list_to_sentences(aligned_dev)]

# %%
for input_item, label_item in zip(inputs_dev, to_zip_dev):
    input_item['labels'] = label_item

# Print the updated inputs list
print(inputs_dev[0])

# %%
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# %%
#!pip install evaluate

# %%
#!pip install seqeval 

# %%
import evaluate

metric = evaluate.load("seqeval")

# %%
id2label = {v: k for k, v in label2id.items()}

# %%
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label = id2label
)

# %%
model.config.num_labels

# %%
import numpy as np


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

# %%
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Convert labels to a list of lists if it's a set
    if isinstance(labels, set):
        labels = [labels]

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


# %%
#!pip install accelerate -U

# %%
from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    push_to_hub=False,
)

# %%


# %%
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=inputs,
    eval_dataset=inputs_dev,
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()

# %%
#import pickle

# Get the trained model
#trained_model = trainer.model

# Save the trained model as a pickle file
#with open("trained_model.pkl", "wb") as f:
    #pickle.dump(trained_model, f)


# %%
inputs_test = []
for sentence in test_sent:
    inputt = tokenizer(sentence, is_split_into_words=True)
    inputs_test.append(inputt)

# %%
test_labels_num = [label2id.get(label, label2id[UNK]) for label in test_flat_labels]

# %%
word_ids = []
for i in inputs_test:
    word_ids.append(i.word_ids())
word_ids_flat = sum(word_ids, [])
aligned_test = align_labels_with_tokens(test_labels_num, word_ids_flat)

# %%
to_zip_test = [[-100] + sublist + [-100] for sublist in list_to_sentences(aligned_test)]

# %%
for input_item, label_item in zip(inputs_test, to_zip_test):
    input_item['labels'] = label_item

# %%
import pickle

with open("trained_model.pkl", "rb") as f:
    trained_model = pickle.load(f)

# %%
if isinstance(trained_model, AutoModelForTokenClassification):
    print("Trained model loaded successfully!")
else:
    print("Error: Failed to load the trained model.")

# %%
import pickle
from transformers import AutoModelForTokenClassification

try:
    # Open the pickle file for reading
    with open("trained_model.pkl", "rb") as f:
        # Deserialize the trained model object
        loaded_model = pickle.load(f)

    # Check if the loaded object is an instance of AutoModelForTokenClassification
    if isinstance(loaded_model, AutoModelForTokenClassification):
        print("Trained model loaded successfully!")
    else:
        print("Error: Loaded object is not an instance of AutoModelForTokenClassification.")
except Exception as e:
    print("Error occurred while loading the trained model:", e)


# %%
inputs_dev[:10]

# %%
inputs_test

# %%


# %%
trainer.model.to('cpu')

# %%


# %%


# %%


# %%
import torch

# List of sentences
sentences = test_flat_sent

# Tokenize the sentences
tokenized_inputs = tokenizer(sentences, truncation=True, padding=True, return_tensors="pt")

# Get the input tensors
input_ids = tokenized_inputs["input_ids"]
attention_mask = tokenized_inputs["attention_mask"]

# Set the model to evaluation mode
trainer.model.eval()

# Define batch size
batch_size = 16  # You can adjust this as needed


# %%
print(input_ids.device.type)
print(attention_mask.device.type)

# %%
import torch

# List of sentences
sentences = test_flat_sent

# Tokenize the sentences
tokenized_inputs = tokenizer(sentences, truncation=True, padding=True, return_tensors="pt")

# Get the input tensors
input_ids = tokenized_inputs["input_ids"]
attention_mask = tokenized_inputs["attention_mask"]

# Set the model to evaluation mode
trainer.model.eval()

# Define batch size
batch_size = 16  # You can adjust this as needed

# Batch processing
with torch.no_grad():
    predicted_labels = []
    for i in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size]

        # Forward pass
        outputs = trainer.model(batch_input_ids, attention_mask=batch_attention_mask)

        # Get the predicted labels (class indices)
        batch_predicted_labels = torch.argmax(outputs.logits, dim=-1)

        predicted_labels.extend(batch_predicted_labels.tolist())

# Process the predictions as needed
# For example, convert logits to labels, post-process the output, etc.

# Print or use the predictions
print(predicted_labels)


# %%
trainer.model

# %%



