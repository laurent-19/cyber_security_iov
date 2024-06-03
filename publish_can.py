import socket
import pandas as pd
import time
import struct

# Load test data
test_data = pd.read_csv('data/test_data.csv')

# Initialize UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('127.0.0.1', 10000)  # 10000 is the destination port

# Function to create a CAN-like packet
def create_can_packet(row):
    can_id = int(row['ID'])
    data_length = 8  # CAN frame standard data length
    data = [int(row[f'DATA_{i}']) for i in range(8)]
    # Create a CAN-like packet with a recognizable format
    can_packet = struct.pack('>IB3x8B', can_id, data_length, *data)
    return can_packet

print('Sending data...')
# Send data
for _, row in test_data.iterrows():
    message = create_can_packet(row)
    sock.sendto(message, server_address)
    time.sleep(0.0005)  # Adjust the sleep time as needed to mimic CAN bus transmission rate

sock.close()

print('Sent all data.')
