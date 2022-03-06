import json
import numpy as np
import pandas as pd
import re
import csv
import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from model import NeuralNet, AdvancedNeuralNet
from transformers import AutoModel, BertTokenizerFast
from torchinfo import summary
from nltk_utils import tokenize, stem, bag_of_words
from sklearn.preprocessing import LabelEncoder

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Add flattened JSON data to an array to store in a dataframe
rows = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
        # Store each pattern with its tag
        pattern = re.sub(r'[^a-zA-Z ]+', '', pattern)
        rows.append([pattern, intent['tag']])

intentsCSV = "intents.csv"
# Write flattened JSON data to CSV file
with open(intentsCSV, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['text', 'label'])
    csvwriter.writerows(rows)

# Convert CSV into dataframe
df = pd.read_csv(intentsCSV)

# Converting the labels into encodings
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
train_text, train_labels = df['text'], df['label']


print(all_words)
ignore_words = ['?', '!', '.', ',']
#We don't want punctuation marks
all_words = [stem(w) for w in all_words if w not in ignore_words]

print("---------------")
print("All our words after tokenization")
print(all_words)

all_words = sorted(set(all_words))
tags = sorted(set(tags))

#Now we are creating the lists to train our data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

#Convert into a numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)

# Import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased')
# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# tokenize and encode sequences in the training set
tokens_train = tokenizer(
    train_text.tolist(),
    max_length=6,
    # Hyperparameter tuning #4: max_length=5,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())


# #Create a new Dataset
# class ChatDataset(Dataset):
#     def __init__(self):
#         self.n_samples = len(X_train)
#         self.x_data = X_train
#         self.y_data = y_train

#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     def __len__(self):
#         return self.n_samples

#DONE: How do these hyperparameters affect optimization of our chatbot? 
"""
The batch size refers to the number of training data samples propagated through 
one forward or backward pass of the model. Smaller batch sizes require less memory, 
thereby saving computational resources. However, smaller batch sizes correspond to
less accurate estimates of the error gradient. 

In this model, the hidden size is the number of nodes in a hidden layer. A general 
rule-of-thumb for hidden layers is to set the number of nodes to somewhere between 
the number of nodes in the input layer and the number of nodes in the output layer.

The output size is the number of nodes in the output layer. There are 7 tags, which 
are essentially classes for this model. In a classification task, the number of output 
nodes should be equal to the number of classes. Thus, the output size should be 7 
(and should not be tuned).

The learning rate refers to the step size in the gradient descent algorithm. This 
step size is the amount by which weights are updated during backpropagation. A high 
learning rate may converge to suboptimal weights (as the loss might settle in a local 
minimum), while a small learning rate may not sufficiently update the weight values.

The number of epochs is the number of times that the model runs through all of the 
training data. Higher learning rates generally require fewer epochs, while lower 
learning rates typically require more epochs. 
"""
batch_size = 8
# Hyperparameter tuning #3: batch_size = 11
hidden_size = 8
# Hyperparameter tuning #2: hidden_size = 26
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000
# Hyperparameter tuning #1: num_epochs = 700


input_size = len(X_train[0])
print("Below is the Input Size of our Neural Network")
print(input_size, len(all_words))
print("Below is the output size of our neural network, which should match the amount of tags ")
print(output_size, tags)

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# DataLoader for train set
train_loader = DataLoader(
    train_data, sampler=train_sampler, batch_size=batch_size)
#The below function helps push to GPU for training if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = NeuralNet(input_size, hidden_size, output_size).to(device)

# freeze all the parameters. This will prevent updating of model weights during fine-tuning.
for param in bert.parameters():
    param.requires_grad = False

model = AdvancedNeuralNet(bert, input_size, hidden_size, output_size).to(device)
summary(model)

#Loss and Optimizer

#DONE: Experiment with another optimizer and note any differences in loss of our model. Does the final loss increase or decrease?
"""
The original Adam optimizer yielded a final loss of 0.0006, which is excellent. 
The AdamW optimizer, on the other hand, yielded a final loss of 0.0008. 
I think this difference is basically negligible, as the values are very close. 
Although the final loss increased with the new optimizer, we can't really compare them 
without running several more trials. 
""" 
#DONE CONT: Speculate on why your changed optimizer may increase or decrease final loss
"""
AdamW is a modified version of the Adam optimizer that includes a revised implementation 
of weight decacy (https://www.fast.ai/2018/07/02/adam-weight-decay/). The results from 
the original paper showed that Adam and AdamW performed very similarly, but AdamW was 
designed to fix Adam in certain edge cases. Since the two optimizers are so similar, 
that is why they produced similar results here.
"""
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch

        #Forward pass
        outputs = model(sent_id, mask)
        loss = criterion(outputs, labels)

        #backward and optimizer step 
        optimizer.zero_grad()

        #Calculate the backpropagation
        loss.backward()
        optimizer.step()

    #Print progress of epochs and loss for every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

#Need to save the data 
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete, file saved to {FILE}')
#Should save our training data to a pytorch file called "data"