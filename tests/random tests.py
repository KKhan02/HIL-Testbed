import serial
import time
import sys

PORT = "COM3"
BAUD = 115200  # Matched to your .ino file's #define BAUD_RATE 115200

def run_diagnostics():
    print(f"=== HIL Testbed Serial Diagnostic ===")
    print(f"Opening {PORT} at {BAUD} baud...")
    
    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.1)
    except Exception as e:
        print(f"Failed to open port: {e}")
        sys.exit(1)

    ser.dtr = True

    print("\n[PHASE 1: Boot sequence & READY signal]")
    print("Listening for 3 seconds...")
    start_time = time.time()
    
    boot_data = b""
    while time.time() - start_time < 3.0:
        if ser.in_waiting:
            chunk = ser.read(ser.in_waiting)
            boot_data += chunk
            
    print(f"Raw Boot Output: {boot_data}")
    if b"READY" in boot_data:
        print("-> SUCCESS: 'READY' detected.")
    else:
        print("-> WARNING: 'READY' not detected. Is the baud rate correct?")

    print("\n[PHASE 2: The Flush Test]")
    print("Sending a single '\\n' to see if the Arduino has garbage in its buffer...")
    ser.write(b'\n')
    time.sleep(0.5)
    
    flush_resp = b""
    while ser.in_waiting:
        flush_resp += ser.read(ser.in_waiting)
        
    print(f"Raw Flush Response: {flush_resp}")
    if b"ERR:UNKNOWN" in flush_resp:
        print("-> BINGO: The Arduino replied ERR:UNKNOWN to an empty newline.")
        print("   This proves Windows is sending a glitch character upon connection.")

    print("\n[PHASE 3: The INIT Handshake]")
    cmd = b"INIT:105\n"
    print(f"Sending exactly: {cmd}")
    ser.write(cmd)
    time.sleep(0.5)
    
    init_resp = b""
    while ser.in_waiting:
        init_resp += ser.read(ser.in_waiting)
        
    print(f"Raw INIT Response: {init_resp}")
    if b"ACK:INIT" in init_resp:
        print("-> SUCCESS: Arduino accepted the INIT handshake.")
    else:
        print("-> FAILED: Arduino did not send ACK:INIT.")

    ser.close()
    print("\n=== Diagnostics Complete ===")

if __name__ == "__main__":
    run_diagnostics()