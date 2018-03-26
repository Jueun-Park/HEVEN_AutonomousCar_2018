import communication

def communicationstart() :
    port = 'COM5'
    # e.g. /dev/ttyUSB0 on GNU/Linux or COM3 on Windows.
    platform = communication.PlatformSerial(port)

    while True:
        platform.test_communication_main()