import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import datetime as dt
import math
import matplotlib.pyplot as plt

# Path to the dataset
path=r'C:\Users\U357407\OneDrive - IBERDROLA S.A\14. Education\05. Stanford\01. Artificial Intelligence\05. CS236 - Deep Generative Models\Papers and Researchs - Project Support\Datasets\TexasTurbine.csv'

# Specify data types for efficient loading
dtypes={'Time stamp': np.object_,'System power generated | (kW)':np.float16, 'Wind speed | (m/s)':np.float16, 'Wind direction | (deg)':np.float16, 'Pressure | (atm)':np.float16 ,'Air temperature | (C)': np.float16}

# Load and preprocess the data
data = pd.read_csv(path, dtype=dtypes)
data['Time stamp'] = pd.to_datetime(data['Time stamp'], format='%b %d, %I:%M %p').dt.strftime('%b %d %H:%M:%S.%f')
data=data.set_index(data['Time stamp'])
data['Time stamp'] = data['Time stamp'].apply(lambda x: dt.datetime.strptime(x,'%b %d %H:%M:%S.%f') if type(x)==str else pd.NaT)
data['Month'] = data["Time stamp"].dt.month
dateNow=dt.date.today()

# Split the data into training and testing sets
train_set = data[:8000]
test_set = data[8000:]

# Define the columns to be scaled
columns_to_scale=['System power generated | (kW)', 'Wind speed | (m/s)','Wind direction | (deg)','Pressure | (atm)', 'Air temperature | (C)']

# Initialize the scaler and scale the data
scaler = MinMaxScaler() #StandardScaler()
scaler = scaler.fit(data[columns_to_scale].values)
test_data=torch.zeros(len(test_set),5)
train_data = torch.zeros((len(train_set), 5))

# Scale train and test data
for i, column in enumerate(columns_to_scale):
    train_data[:, i] = torch.from_numpy(train_set[column].values)
    test_data[:, i] = torch.tensor(test_set[column].values, dtype=torch.float32)

train_data=scaler.transform(train_data)
test_data=scaler.transform(test_data)

# Prepare the data for training the model
train_labels = torch.zeros(len(train_set))
train_set = [(train_data[i], train_labels[i]) for i in range(len(train_set))]
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

# Set a manual seed for reproducibility
torch.manual_seed(77)

# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 5)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define the Discriminator class
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers): #self, input_size, hidden_size, num_layers
        super(Discriminator,self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x,_ = self.lstm(x)
        output = self.model(x)
        return output

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Generator,self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        # Pass the input through the LSTM
        x, _ = self.lstm(x)
        output = self.model(x)
        return output

# Initialize the models    
discriminator = Discriminator(input_size=16, hidden_size=16, num_layers=3)
generator = Generator(input_size=5, hidden_size=128, num_layers=5,output_size=5)
autoencoder = Autoencoder()

# Set learning rates and loss functions
lr = 0.001
num_epochs = 300
loss_function = nn.BCELoss()
criterion= nn.MSELoss()
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_autoencoder = torch.optim.Adam(autoencoder.parameters(), lr=lr)

# Lists to store losses
generator_losses = []
discriminator_losses = []

# Training loop for GAN
for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        
        real_samples = real_samples.float()
        # Data for training the discriminator
        real_samples_labels = torch.ones((32, 1))
        latent_space_samples = torch.randn((32, 5))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((32, 1))
        
        all_samples = torch.cat((real_samples, generated_samples))
        
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)

        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((32, 5))

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

        # Store losses
        generator_losses.append(loss_generator.item())
        discriminator_losses.append(loss_discriminator.item())
        
        # Show loss
        if n == 32 - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
        elif epoch %100==0:
            torch.save(generator.state_dict(), f'generator_model_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'discriminator_model_epoch_{epoch + 1}.pth')



for epoch in range(200):
    for n, (real_samples, _) in enumerate(train_loader):
        real_samples = real_samples.float()
        optimizer_autoencoder.zero_grad()
        outputs = autoencoder(real_samples)
        loss = criterion(outputs, real_samples)
        loss.backward()
        optimizer_autoencoder.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

threshold = 0.15 

# Plotting the training losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(generator_losses, label="Generator", color='blue')
plt.plot(discriminator_losses, label="Discriminator", color='red')
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

num_samples = 1000  # number of samples you want to generate
latent_space_samples = torch.randn((num_samples, 5))  # adjust the size according to your model
test_data = torch.tensor(test_data, dtype=torch.float32)
generated_data = generator(latent_space_samples).detach().numpy()
generated_data=scaler.inverse_transform(generated_data)
generated_power_output = generated_data

generated_data_tensor = torch.tensor(generated_power_output, dtype=torch.float32)
reconstructed_data = autoencoder(generated_data_tensor)
reconstruction_error = criterion(reconstructed_data, generated_data_tensor)

# Detect anomalies
anomalies = reconstruction_error > threshold

anomaly_data = generated_data[anomalies]

test_plot_data=test_data
test_plot_data=scaler.inverse_transform(test_data)

plt.figure(figsize=(10, 5))
plt.title("Anomaly Data")
plt.plot(anomaly_data, label="Anomally Signal", color='red')
#plt.plot(generated_data, label="Generated Data", color='blue')
plt.xlabel("Time")
plt.ylabel("Output")
plt.legend()
plt.show()

df1=pd.DataFrame(generated_power_output)
df1.to_csv(f'generated_data_{dateNow}.csv')
df2=pd.DataFrame(test_data)
df2.to_csv(f'latent_data_{dateNow}.csv')


plt.figure(figsize=(12, 6))
plt.plot(test_plot_data)
plt.title('Test Data Distribution')
plt.xlabel('System Power Generated (kW)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(generated_power_output)
plt.title('Generated Data Distribution')
plt.xlabel('System Power Generated (kW)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
