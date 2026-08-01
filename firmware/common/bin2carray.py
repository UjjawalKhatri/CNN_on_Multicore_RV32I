import sys
if len(sys.argv) != 3:
    print("Usage: python3 bin2carray.py in.bin out.hex"); sys.exit(1)
data = open(sys.argv[1], "rb").read()
# pad to 4 bytes
pad = (-len(data)) % 4
data += b"\x00"*pad
with open(sys.argv[2], "w") as out:
    for i in range(0, len(data), 4):
        w = int.from_bytes(data[i:i+4], "little")
        out.write(f"0x{w:08X},\n")
