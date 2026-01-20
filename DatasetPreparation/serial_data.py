import serial 
import pandas as pd


# Configure serial port parameters
port = 'COM8'  # /////////////////////////////////////////////////////////////// Replace with your ESP32's COM port
baud_rate = 115200  # /////////////////////////////////////////////////////////// Replace with your ESP32's baud rate

data_x = 'data_x.csv'#/////////////////////////////////////////////////////////// Replace with your motion data file path
data_y = 'data_y.csv'#/////////////////////////////////////////////////////////// Replace with your action label file path (e.g., if recording action 1, it will store index 0)
label = [2] #//////////////////////////////////////////////////////////////////// 0 - scissors 1 - rock 2 - paper

num = 0

try:
    # Open serial port
    ser = serial.Serial(port, baud_rate)
    print(f"Port {port} opened successfully, baud_rate {baud_rate}")

    # Open or create CSV files
    with open(data_x, mode='a', newline='') as file1, open(data_y, mode='a', newline='') as file2:
        while True:
            # Read serial data
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                print(f"Data: {line}")  # Print received data

                try:
                    # Split data by commas and convert to float32
                    data = [float(x) for x in line.split(',')]

                    # Convert data to DataFrames
                    df_x = pd.DataFrame([data])
                    df_y = pd.DataFrame([label])

                    # Append data to CSV files
                    df_x.to_csv(file1, header=False, index=False)
                    df_y.to_csv(file2, header=False, index=False)

                    # Flush file buffers
                    file1.flush()
                    file2.flush() 
                    num += 1
                    print(f"Data appended to CSV files successfully. Count: {num}")
                   

                except ValueError as e:
                    print(f"Data conversion error: {e}")

except serial.SerialException as e:
    print(f"Failed to open serial port: {e}")
except Exception as e:
    print(f"Error occurred: {e}")