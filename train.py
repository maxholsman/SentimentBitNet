import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from dataset import AmazonReviewDataset
from model import SentimentTransformer, build_sentiment_transfomer
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

def read_amazon_reviews(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 512
    batch_size = 32
    
    file_path = '/Users/maximilianholsman/Library/CloudStorage/OneDrive-Personal/Documents/Work/Fall 2023/Projects/Bit_Transformer/train.ft.txt'
    
    raw_data = read_amazon_reviews(file_path)
    
    labels = [1 if line.split(' ')[0] == '__label__2' else 0 for line in raw_data]
    
    reviews = [' '.join(line.split(' ')[1:]) for line in raw_data]

    df = pd.DataFrame(list(zip(labels, reviews)), columns=['label', 'review'])

    print("Creating dataset and dataloader...")
    dataset = AmazonReviewDataset(
        reviews=df['review'].to_numpy(),
        labels=df['label'].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    data_loader = DataLoader(dataset, batch_size=batch_size)
    
    vocab_size = tokenizer.vocab_size
    
    #d_model, vocab_size, seq_len, num_heads, d_ff, num_layers, dropout=0.1, num_classes=2
    
    print("Creating model...")
    model = build_sentiment_transfomer(512, vocab_size, 512, 8, 2048, 6, 0.1, 2)
    model = model.to(device)
    
    num_epochs = 10
    lr = 2e-3
    optimzer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    
    print('Starting training...')
    for epoch in range(num_epochs):
        for i, batch in enumerate(data_loader):
            print(f'Epoch: {epoch}, Batch: {i}')
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            #forward pass
            outputs = model(inputs, attention_mask)
            loss = criterion(outputs, labels)
            
            #backward pass
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
            
train()
    
    

