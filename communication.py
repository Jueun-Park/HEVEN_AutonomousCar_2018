# 통신 프로그램
# 제어에서 받은 정보로 통신 패킷 만들어서 플랫폼으로 보내기
# 플랫폼에서 통신 패킷 받아와서 제어로 보내기
# 패킷 세부 형식(string)은 책자 참조
# input: (from car_control)
# output: (to car_control)

import serial
import socket
