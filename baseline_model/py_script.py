# %%
def read_sent(path):
    ents = []
    curEnts = []
    for line in open(path):
        line = line.strip()
        if line == '':
            ents.append(curEnts)
            curEnts = []
        elif line[0] == '#' and len(line.split('\t')) == 1:
            continue
        else:
            curEnts.append(line.split('\t')[1])
    return(ents)

def read_labels(path):
    ents = []
    curEnts = []
    for line in open(path):
        line = line.strip()
        if line == '':
            ents.append(curEnts)
            curEnts = []
        elif line[0] == '#' and len(line.split('\t')) == 1:
            continue
        else:
            curEnts.append(line.split('\t')[2])
    return(ents)

def read_index(path):
    ents = []
    curEnts = []
    for line in open(path):
        line = line.strip()
        if line == '':
            ents.append(curEnts)
            curEnts = []
        elif line[0] == '#' and len(line.split('\t')) == 1:
            continue
        else:
            curEnts.append(line.split('\t')[0])
    return(ents)

# %%
print('test')

# %%
#Training data

#returns list of lists
training_labels = read_labels("en_ewt-ud-train.iob2")
training_sent = read_sent("en_ewt-ud-train.iob2")

#flatten to one list to be able to use myutils
train_labels = sum(training_labels, [])
train_sent = sum(training_sent, [])

# %%
#Evaluation data

dev_labels = read_labels("en_ewt-ud-dev.iob2")
dev_sent = read_sent("en_ewt-ud-dev.iob2")

dev_flat_labels = sum(dev_labels, [])
dev_flat_sent = sum(dev_sent, [])

# %%
#Test data
#Keeping track of indeces to save to required .iob2 format for model's predictions

test_labels = read_labels("en_ewt-ud-test.iob2")
test_sent = read_sent("en_ewt-ud-test.iob2")
test_index = read_index("en_ewt-ud-test.iob2")

test_flat_labels = sum(test_labels, [])
test_flat_sent = sum(test_sent, [])
test_flat_index = sum(test_index, [])

# %%
#!pip install transformers

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
training_labels[0]


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
import myutils
UNK = "[UNK]"

id2label, label2id = myutils.labels2lookup(train_labels, UNK)
NLABELS = len(id2label)
print(train_labels)
print(label2id)

#converting BIO labels to numerical labels
train_labels_num = [label2id.get(label, label2id[UNK]) for label in train_labels]

# %%
labels = training_labels
word_ids = []
for i in inputs:
    word_ids.append(i.word_ids())
word_ids_flat = sum(word_ids, [])
aligned_training = align_labels_with_tokens(train_labels_num, word_ids_flat)


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
dev_ner_labels = [label2id.get(label, label2id[UNK]) for label in dev_flat_labels]

# %%
word_ids = []
for i in inputs_dev:
    word_ids.append(i.word_ids())
word_ids_flat = sum(word_ids, [])
aligned_dev = align_labels_with_tokens(dev_ner_labels, word_ids_flat)

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
    num_train_epochs=3,
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
# labels = raw_datasets["train"][0]["ner_tags"]
# labels = [label_names[i] for i in labels]


# # %%
# import torch


# # List of sentences
# sentences = test_flat_sent

# # Tokenize the sentences
# tokenized_inputs = tokenizer(sentences, truncation=True, padding=True, return_tensors="pt")

# # Get the input tensors
# input_ids = tokenized_inputs["input_ids"]
# attention_mask = tokenized_inputs["attention_mask"]

# # Set the model to evaluation mode
# trainer.model.eval()
# # Make predictions
# with torch.no_grad():
#     # Forward pass
#     outputs = trainer.model(input_ids, attention_mask=attention_mask)

#     # Get the predicted labels (class indices)
#     predicted_labels = torch.argmax(outputs.logits, dim=-1)



# # Process the predictions as needed
# # For example, convert logits to labels, post-process the output, etc.

# # Print or use the predictions
# print(predicted_labels)

trainer.model.to('cpu')

import torch
from transformers import Trainer

# Define a function for batch processing
def batch_process(sentences, tokenizer, trainer):
    # Tokenize the sentences
    tokenized_inputs = tokenizer(sentences, truncation=True, padding=True, return_tensors="pt")
    # Get the input tensors
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]
    # Set the model to evaluation mode
    trainer.model.eval()
    # Make predictions
    with torch.no_grad():
        # Forward pass
        outputs = trainer.model(input_ids, attention_mask=attention_mask)
        # Get the predicted labels (class indices)
        predicted_labels = torch.argmax(outputs.logits, dim=-1)
    # Process the predictions as needed
    # For example, convert logits to labels, post-process the output, etc.
    return predicted_labels


# Define the batch size
batch_size = 8

# Split the list of sentences into batches
batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]

# Process each batch and print or use the predictions
for batch in batches:
    predictions = batch_process(batch, tokenizer, trainer)
    print(predictions)

# %%
trainer.model

# %%



