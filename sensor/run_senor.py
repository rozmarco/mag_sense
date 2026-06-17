import serial
import struct
import time

# Change this to your actual COM port:
# Windows: 'COM3', 'COM4', etc.
# Mac/Linux: '/dev/ttyUSB0' or '/dev/tty.usbserial-...'
PORT = '/dev/ttyUSB0'
BAUD = 9600  # WT901 default baud rate

def parse_packet(data):
    """Parse a single 11-byte WT901 data packet."""
    if len(data) < 11 or data[0] != 0x55:
        return None
    
    packet_type = data[1]
    
    if packet_type == 0x51:  # Acceleration
        ax, ay, az = struct.unpack('<hhh', data[2:8])
        return {
            'type': 'accel',
            'ax': ax / 32768.0 * 16,  # g
            'ay': ay / 32768.0 * 16,
            'az': az / 32768.0 * 16,
        }
    elif packet_type == 0x52:  # Gyroscope
        gx, gy, gz = struct.unpack('<hhh', data[2:8])
        return {
            'type': 'gyro',
            'gx': gx / 32768.0 * 2000,  # deg/s
            'gy': gy / 32768.0 * 2000,
            'gz': gz / 32768.0 * 2000,
        }
    elif packet_type == 0x53:  # Angle
        roll, pitch, yaw = struct.unpack('<hhh', data[2:8])
        return {
            'type': 'angle',
            'roll':  roll  / 32768.0 * 180,  # degrees
            'pitch': pitch / 32768.0 * 180,
            'yaw':   yaw   / 32768.0 * 180,
        }
    return None

def read_wt901(port=PORT, baud=BAUD):
    ser = serial.Serial(port, baud, timeout=1)
    print(f"Connected to {port} at {baud} baud")
    
    buffer = bytearray()
    
    try:
        while True:
            raw = ser.read(11)
            if not raw:
                continue
            
            buffer.extend(raw)
            
            # Find packet start (0x55) and extract 11-byte packets
            while len(buffer) >= 11:
                if buffer[0] == 0x55:
                    packet = bytes(buffer[:11])
                    result = parse_packet(packet)
                    if result:
                        if result['type'] == 'accel':
                            print(f"Accel (g):  X={result['ax']:.3f}  Y={result['ay']:.3f}  Z={result['az']:.3f}")
                        elif result['type'] == 'gyro':
                            print(f"Gyro (°/s): X={result['gx']:.2f}  Y={result['gy']:.2f}  Z={result['gz']:.2f}")
                        elif result['type'] == 'angle':
                            print(f"Angle (°):  Roll={result['roll']:.2f}  Pitch={result['pitch']:.2f}  Yaw={result['yaw']:.2f}")
                    buffer = buffer[11:]
                else:
                    buffer.pop(0)  # Discard byte until we find 0x55
                    
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()

if __name__ == '__main__':
    read_wt901()
