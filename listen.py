import socket
import torch
import pandas as pd
from torch import nn
import struct
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler("log/can_detection_infer.log")
])

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load normalization parameters from 'train_data.csv'
train_data = pd.read_csv('data/train_data.csv')

# Extract min and max values for normalization
norm_params = {}
feature_columns = ['ID'] + [f'DATA_{i}' for i in range(8)]
for column in feature_columns:
    min_val = train_data[column].min()
    max_val = train_data[column].max()
    norm_params[column] = (min_val, max_val)

# Define the model with updated layer names
class MultiTaskNN(nn.Module):
    def __init__(self, input_size, num_labels, num_categories, num_specific_classes):
        super(MultiTaskNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output1 = nn.Linear(64, num_labels)
        self.output2 = nn.Linear(64, num_categories)
        self.output3 = nn.Linear(64, num_specific_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        label_out = self.output1(x)
        category_out = self.output2(x)
        specific_class_out = self.output3(x)
        return label_out, category_out, specific_class_out

# Load model
input_size = len(['ID'] + [f'DATA_{i}' for i in range(8)])
num_labels = 2  # Benign, Attack
num_categories = 3  # Benign, DoS, Spoofing
num_specific_classes = 6  # Benign, Speed, DoS, Gas, RPM, Steering Wheel

model = MultiTaskNN(input_size, num_labels, num_categories, num_specific_classes)
model.load_state_dict(torch.load('models/iov_model_epoch_8_loss_0.0000.pth'))
model.to(device)  # Move model to the appropriate device (GPU or CPU)
model.eval()

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('127.0.0.1', 10000)
sock.bind(server_address)

# Calculate average inference time per specific class
specific_class_times = {
    'Benign': [],
    'DoS': [],
    'Gas': [],
    'RPM': [],
    'Speed': [],
    'Steering Wheel': []
    }

# Function to preprocess incoming data
def preprocess_data(can_id, data):
    df = pd.DataFrame([[can_id] + data], columns=['ID'] + [f'DATA_{i}' for i in range(8)])
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Normalize using extracted parameters
    for column in df.columns:
        min_val, max_val = norm_params[column]
        if min_val == max_val:
            df[column] = 0.5  # Assign a constant value if min and max are the same
        else:
            df[column] = (df[column] - min_val) / (max_val - min_val)
    
    return torch.tensor(df.values, dtype=torch.float32).to(device)  # Move tensor to the appropriate device

# Listen for incoming data
while True:
    data, address = sock.recvfrom(4096)
    if data:
        # Unpack the CAN-like packet
        unpacked_data = struct.unpack('>IB3x8B', data)
        can_id = unpacked_data[0]
        data_bytes = list(unpacked_data[2:])
        
        features = preprocess_data(can_id, data_bytes)
        
        with torch.no_grad():
            start_time = time.time()
            label_out, category_out, specific_class_out = model(features)
            end_time = time.time()

            _, label_pred = torch.max(label_out, 1)
            _, category_pred = torch.max(category_out, 1)
            _, specific_class_pred = torch.max(specific_class_out, 1)

            inference_time = (end_time - start_time) * 1e3 # Convert to miliseconds
        
        if torch.isnan(label_out).any() or torch.isnan(category_out).any() or torch.isnan(specific_class_out).any():
            logging.error("NaN values found in model output")
        
        label = 'Benign' if label_pred.item() == 1 else 'Attack'
        category = ['Benign', 'DoS', 'Spoofing'][category_pred.item()]
        specific_class = ['Benign', 'DoS', 'Gas', 'RPM', 'Speed', 'Steering Wheel'][specific_class_pred.item()]
        
        if label == 'Attack':  
            # Print the CAN data
            logging.info(f"Received CAN data - ID: {can_id}, Data: {data_bytes}")
            # Print the detected label, category, and specific class
            logging.info(f"Detected: {label}, Category: {category}, Specific Class: {specific_class}")
            # Print the inference time
            logging.info(f"Inference Time: {inference_time:.6f} ms")
            
        specific_class_times[specific_class].append(inference_time)
            
        for specific_class, times in specific_class_times.items():
            if len(times) % 100 == 0 and len(times) > 0:
                avg_time = sum(times) / len(times)
                logging.info(f"Average Inference Time for {specific_class}: {avg_time:.6f} ms (n={len(times)})")