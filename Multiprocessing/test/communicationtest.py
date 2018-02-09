import communication
from communication import PlatformSerial
if __name__ == '__main__':
    port = 'COM3'
    # e.g. /dev/ttyUSB0 on GNU/Linux or COM3 on Windows.
    platform = PlatformSerial(port)
    print('CONNECTED')

    while True:
        platform.test_communication_main()