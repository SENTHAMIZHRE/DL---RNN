# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset



## DESIGN STEPS
### STEP 1:

Import the required libraries such as NumPy, Pandas, Matplotlib, and TensorFlow/Keras. Load the historical stock dataset and display the data.

### STEP 2:

Select the Closing Price column from the dataset and preprocess the data. Normalize the closing prices using MinMaxScaler so that the values are between 0 and 1.

### STEP 3:

Create input sequences using previous stock prices. For example, use the previous 60 days of closing prices to predict the closing price of the next day.

### STEP 4:

Split the prepared dataset into training and testing data. Reshape the input data into the format required by the RNN model.

### STEP 5:

Build and train the RNN model using SimpleRNN layers along with Dense layers. Compile the model using an appropriate optimizer and loss function, and train it using the training dataset.

### STEP 6:

Use the trained model to predict stock prices for the test data. Convert the predicted values back to the original scale and compare the actual and predicted prices using a graph. Evaluate the model's prediction performance.




## PROGRAM

### Name: SENTHAMIZH SELVAN K

### Register Number:212223235001

```python
%pip install torchinfo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torchinfo import summary
from torch.utils.data import DataLoader, TensorDataset

df_train = pd.read_csv(r"C:\Users\admin\Downloads\trainset.csv")
df_test = pd.read_csv(r"C:\Users\admin\Downloads\testset.csv")
train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)
x_train.shape, y_train.shape, x_test.shape, y_test.shape
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
class RNNModel(nn.Module):
  def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self,x):
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])
    return out

model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
summary(model, input_size=(64, 60, 1))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 20
model.train()
train_losses = []
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss / len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}], Loss:{train_losses[-1]:.4f}")
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()
    
# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')
```

### OUTPUT

## Training Loss Over Epochs Plot
<img width="518" height="508" alt="image" src="https://github.com/user-attachments/assets/3383be37-10f6-416f-9b7d-8c233ed17a32" />



## True Stock Price, Predicted Stock Price vs time
<img width="717" height="552" alt="image" src="https://github.com/user-attachments/assets/df1e49fe-77f5-4342-b902-9e74349be25d" />


### Predictions
<img width="778" height="482" alt="image" src="https://github.com/user-attachments/assets/e40079f0-5a11-4b80-969e-ec7737e7e0e2" />

## RESULT
The RNN model was successfully developed and trained to predict stock prices using historical closing price data. The predicted and actual prices were compared to evaluate the model performance.
