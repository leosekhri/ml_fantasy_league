
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split

# Assuming you have already loaded your data from CSVs into pandas DataFrames

df_2023_2024 = pd.read_csv('2023-2024.csv')
df_2020_2021 = pd.read_csv('2020-2021.csv')
df_2022_2023= pd.read_csv('2022-2023.csv')
selective_df_2023_2024=df_2023_2024[['assists', 'goals_conceded', 'goals_scored','clean_sheets','saves','minutes','yellow_cards','red_cards','total_points']]
selective_df_2022_2023=df_2022_2023[['assists', 'goals_conceded', 'goals_scored','clean_sheets','saves','minutes','yellow_cards','red_cards','total_points']]
selective_df_2022_2023=selective_df_2022_2023.astype(float)
selective_df_2023_2024=selective_df_2023_2024.astype(float)
df_2023_2024=pd.concat([selective_df_2023_2024, selective_df_2022_2023], axis=0)

encoder = OneHotEncoder(sparse_output=False)
a_new_df2 = df_2020_2021[['Assists', 'Goals_Conceded', 'Goals_Scored', 'Clean_Sheets', 'Saves', 'Minutes', 'Yellow_Cards', 'Red_Cards', 'Total_Points']]
a_new_df2['assists'] = df_2020_2021['Assists']
a_new_df2['goals_conceded'] = df_2020_2021['Goals_Conceded']
a_new_df2['goals_scored'] = df_2020_2021['Goals_Scored']
a_new_df2['clean_sheets'] = df_2020_2021['Clean_Sheets']
a_new_df2['saves'] = df_2020_2021['Saves']
a_new_df2['minutes'] = df_2020_2021['Minutes']
a_new_df2['yellow_cards'] = df_2020_2021['Yellow_Cards']
a_new_df2['red_cards'] = df_2020_2021['Red_Cards']
a_new_df2['total_points'] = df_2020_2021['Total_Points']
a_new_df2=a_new_df2.astype(float)

entire_data = pd.concat([df_2023_2024, a_new_df2], axis=0)


X_train_array, X_test_array, y_train_array, y_test_array = train_test_split(
    entire_data[['assists', 'goals_conceded', 'goals_scored','clean_sheets','saves','minutes','yellow_cards','red_cards']].values,
    entire_data[['total_points']].values,
    test_size=0.01,
    random_state=42)


# Convert pandas series to PyTorch tensors
X_train = torch.tensor(X_train_array, dtype=torch.float32)  # Adjust shape as needed
y_train = torch.tensor(y_train_array, dtype=torch.float32)  # Adjust shape as needed
X_test = torch.tensor(X_test_array, dtype=torch.float32)  # Adjust shape as needed
y_test = torch.tensor(y_test_array, dtype=torch.float32)  # Adjust shape as needed

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 1
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(8, 512),  # Adjust input size if needed
            nn.ReLU(),
            nn.Linear(512, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = self.linear_stack(x)
        return x

# Initialize model, loss function, and optimizer
model = NeuralNetwork().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Adjust learning rate as needed

# Training function
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch_idx, (data, targets) in enumerate(dataloader):
        
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        outputs = model(data)
        loss = loss_fn(outputs, targets)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Testing function
def test(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            test_loss += loss_fn(outputs, targets).item() * data.size(0)

    test_loss /= len(dataloader.dataset)
    print(f'Average Test loss: {test_loss:.4f}')
    return test_loss

# Training loop
def fit(min_loss = 20 , patience_epoch = 100 ):
    test_loss=test(test_dataloader, model, loss_fn)
    epochs=1
    while test_loss>= min_loss and epochs < patience_epoch:    
        print("Epoch", epochs)
        epochs=epochs+1
        train(train_dataloader, model, loss_fn, optimizer)
        test_loss=test(test_dataloader, model, loss_fn)

    print("Done!")
fit(24, 150)
