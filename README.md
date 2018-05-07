# HEVEN_AutonomousCar_2018
2018 International Student Car Competition: Autonomous Car SKKU Team. HEVEN

[국제대학생창작자동차경진대회 공식 홈페이지](http://kasa.kr/cev/)

[위키 작성법(마크다운 문서)](https://gist.github.com/ihoneymon/652be052a0727ad59601)

## 자율주행 프로그램 구조
|Category|Role|Program Name|
|:--------|:--------|:--------|
|**Perception**|LiDAR Mapper|```lidar.py```|
|"|Cam Video Stream Control|```video_stream.py```|
|"|Lane Detector|```lane_cam.py```|
|"|Sign Detector|```shape_detection.py```<br>```sign_cam.py```|
|**Planning**|Motion & Prticular Condition Planner|```parabola.py```<br>```motion_planner.py```|
|**Control**|Car Speed/Steering Control|```car_control.py```|
|**Communication**|Communication with Platform|```serial_packet.py```<br>```communication.py```|
|**Process**|Main Process Management|```main.py```|

## 팀원
### 소프트웨어 및 알고리즘 개발 부문
김성우 김윤진 김진웅 김홍빈 박주은 박준혁 예하진 유성룡 이아영 현지웅
### 차량 외장 부문
강희수 김민수 이중구

## 알고리즘
### Perception
1. Vision 데이터 처리
	1. 차선 인식
	2. 표지판 인식
  
2. LiDAR 데이터 처리
	1. 장애물 인식
### Planning
3. 경로 설정
	1. 목표점 설정
	2. 장애물과 차선 회피 경로 설정
	3. 미션 별 경로 계획
### Control
4. 제어
	1. 조향
	2. 속도
### Communication
5. 통신
	1. 전달할 통신 패킷 생성 및 플랫폼 전달
	2. 받은 통신 패킷 해석
### Operation
6. 연산 속도 상승
	1. 프로그램 병렬 처리
	2. CUDA for python
