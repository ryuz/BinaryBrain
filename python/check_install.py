import sys
print(sys.version)

import platform
print("Python        : {}".format(platform.python_version()))

import binarybrain as bb
print("BinaryBrain   : {}".format(bb.get_version_string()))

device_available = bb.is_device_available()
print("GPU available : {}".format(device_available))
if device_available:
    device_count = bb.get_device_count()
    print("GPU count     : {}".format(device_count))
    for i in range(device_count):
        print("GPU[{}]        : {}".format(i, bb.get_device_name(i)))
