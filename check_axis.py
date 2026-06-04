# wt901_probe.py
# Prints all WT901 packet types live so you can physically move
# the sensor/magnet and observe which axes respond to which motion.

import serial
import struct

PORT = '/dev/ttyUSB1'
BAUD = 9600

PACKET_SIZE = 11
HEADER      = 0x55

# Scaling factors from WT901 datasheet
ACC_SCALE  = 16.0 / 32768.0   # g
GYRO_SCALE = 2000.0 / 32768.0 # °/s
ANG_SCALE  = 180.0 / 32768.0  # degrees
MAG_SCALE  = 1.0               # raw counts (no standard unit)

def parse_packet(pkt):
    """
    Parse one 11-byte WT901 packet.
    Returns (type_str, x, y, z) or None if unrecognised type.
    """
    if pkt[0] != HEADER:
        return None

    ptype     = pkt[1]
    x, y, z, _ = struct.unpack('<hhhh', bytes(pkt[2:10]))

    if ptype == 0x51:
        return ('ACC ',
                x * ACC_SCALE,
                y * ACC_SCALE,
                z * ACC_SCALE)

    elif ptype == 0x52:
        return ('GYRO',
                x * GYRO_SCALE,
                y * GYRO_SCALE,
                z * GYRO_SCALE)

    elif ptype == 0x53:
        return ('ANG ',
                x * ANG_SCALE,
                y * ANG_SCALE,
                z * ANG_SCALE)

    elif ptype == 0x54:
        return ('MAG ',
                float(x),
                float(y),
                float(z))

    return None   # 0x50 time packet etc — ignore


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    buf = bytearray()

    print(f"{'Type':6}  {'X':>12}  {'Y':>12}  {'Z':>12}")
    print("-" * 50)

    # Keep one dict of latest values per type so we can
    # print a clean refreshing block rather than scrolling
    latest = {}

    try:
        while True:
            buf.extend(ser.read(11))

            while len(buf) >= PACKET_SIZE:
                # Sync to header
                if buf[0] != HEADER:
                    buf.pop(0)
                    continue

                pkt    = buf[:PACKET_SIZE]
                buf    = buf[PACKET_SIZE:]
                result = parse_packet(pkt)

                if result is None:
                    continue

                ptype, x, y, z = result
                latest[ptype]  = (x, y, z)

                # Reprint all known values in place
                lines = []
                for t, (vx, vy, vz) in latest.items():
                    lines.append(f"{t:6}  {vx:>12.3f}  {vy:>12.3f}  {vz:>12.3f}")

                # Move cursor up and overwrite
                if len(latest) > 1:
                    print(f"\033[{len(latest)}A", end='')
                for line in lines:
                    print(line)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == '__main__':
    main()