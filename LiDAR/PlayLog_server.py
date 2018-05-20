# 라이다 로깅 파일 (.txt)을 라이다 데이터 처리 프로그램이 받을 수 있도록 통신으로 쏴줌
# 2017 팀 헤븐 인원이 개발함


import socket

####################
HOST = ''
PORT = 10012
####################

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print("Socket Created")

LOG_FILE_NAME = "lidar_log_file_name.txt"
f = open(LOG_FILE_NAME, 'r')

client_socket = None

while True:
    if (client_socket is None):
        client_socket, address = server_socket.accept()
        print("Socket Open", address)

    data = f.read(7000)
    try:
        client_socket.send(data)
    except:
        pass

    try:
        if client_socket.recv(1024) == "end":
            print("Shutdown")
            break
        else:
            pass
    except Exception as e:
        print(e)
        break

server_socket.close()
