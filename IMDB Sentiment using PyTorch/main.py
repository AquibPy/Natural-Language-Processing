import warnings
warnings.filterwarnings("ignore")
import torch
import spacy
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator,LabelField
from model import WordLSTM

spacy_eng = spacy.load("en_core_web_sm")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EMB_DIM = 100
HID_DIM = 512
N_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 3e-2
EPOCHS = 100

def tokenize(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

SRC = Field(sequential=True, tokenize=tokenize, use_vocab=True, lower=True)
# TRG = Field(sequential=False, use_vocab=False,dtype= torch.int64)
TRG = LabelField(dtype = torch.int64)
train_data, test_data  = IMDB.splits(SRC,TRG)

# print(f"Number of training examples: {len(train_data.examples)}")
# print(f"Number of testing examples: {len(test_data.examples)}")

# display single example at index 0
# print(vars(train_data.examples[0]))


SRC.build_vocab(train_data,max_size=10000, min_freq = 5, vectors="glove.6B.100d")
TRG.build_vocab(train_data, min_freq = 5)

# print(vars(TRG.vocab))
# print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
# print(f"Unique tokens in TRG vocabulary: {len(TRG.vocab)}")

train_iterator,test_iterator = BucketIterator.splits(
      (train_data, test_data), 
      batch_size = BATCH_SIZE, 
      device = DEVICE
    )

# initializing our model
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
model = WordLSTM(INPUT_DIM, OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=3,verbose=True)
criterion = nn.CrossEntropyLoss()

def train(model,epochs,iterator,optimizer,criterion,clip=1):
    model.train()
    epoch_loss = 0
    total_correct = 0
    total_count = 0
    for epoch in range(1,epochs):
        for i,batch in enumerate(train_iterator):
            src = batch.text.to(DEVICE)
            trg = batch.label.to(DEVICE)
            output = model(src).squeeze(1)
            total_correct += torch.sum(torch.eq(output.argmax(1), trg))
            total_count+=len(trg)
            optimizer.zero_grad()

            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f'correct: {total_correct}/{total_count}')
        mean_loss = epoch_loss / len(iterator)
        scheduler.step(mean_loss)
        print(f'For Epoch {epoch}: Mean_Loss = {mean_loss}')
    torch.save(model.state_dict(),'model.pth')

train(model,EPOCHS,train_iterator,optimizer,criterion,clip=1)

def predict(sentence):

  if type(sentence) == str:
     tokenized_sentence = tokenize(sentence)
  else:
    tokenized_sentence = sentence


  input_data = [SRC.vocab.stoi[word.lower()] for word in tokenized_sentence]
  input_data = torch.tensor(input_data, dtype=torch.int64).unsqueeze(1).to(DEVICE)


  model.eval()
  output = model(input_data)
  predict = output.argmax(1)
  predict = predict.squeeze(0)

  if predict>0:
    return "---->> Positive Review"
  else:
    return '---->> Negative Review'


if __name__=='__main__':
    train(model,EPOCHS,train_iterator,optimizer,criterion,clip=1)
    predict('i have enjoyed this movie')