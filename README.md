# HEVEN_AutonomousCar_2018
2018 International Student Car Competition: Autonomous Car SKKU Team. HEVEN

2018 국제대학생 창작자동차 경진대회 자율주행차 부문 성균관대학교 팀 헤븐

[장려상 수상] [본선 주행 생중계 녹화 영상(1시간 25분 20초부터)](https://tv.naver.com/v/3246704)

* 생중계 영상 끊김이 그대로 녹화되어 영상이 조금씩 끊깁니다.

[국제 대학생 창작자동차 경진대회(International Student Car Competition) 공식 홈페이지](http://kasa.kr/cev/)


## 자율주행 프로그램 구조
|Category|Role|Program Name|Developer|
|:--------|:--------|:--------|:-----------:|
|**Perception**|LiDAR Mapper|```lidar.py```|김홍빈|
|"|Cam Video Stream Control|```video_stream.py```|김진웅|
|"|Lane Detector|```lane_cam.py```|김홍빈 예하진|
|"|Sign Detector|```shape_detection.py```<br>```sign_cam.py```|김윤진 이아영<br>김성우 현지웅|
|**Planning**|Motion & Prticular Condition Planner|```parabola.py```<br>```motion_planner.py```|김홍빈|
|**Control**|Car Speed/Steering Control|```car_control.py```|박준혁|
|**Communication**|Communication with Platform|```serial_packet.py```<br>```communication.py```|김진웅|
|**Process**|Main Process Management|```main.py```|김진웅 유성룡|
* Project Design & Management: 박주은

## 팀원 (2018년)
### 소프트웨어 및 알고리즘 개발 부문 (10)
* 김성우: 시스템경영공학과/컴퓨터공학과
* 김윤진: 전자전기공학부
* 김진웅: 컴퓨터공학과
* 김홍빈: 기계공학부
* 박주은: 시스템경영공학과/인포매틱스
* 박준혁: 전자전기공학부
* 예하진: 공학계열
* 유성룡: 전자전기공학부
* 이아영: 신소재공학부/전자전기공학부
* 현지웅: 소프트웨어학과
### 차량 외장 부문 (3)
* 강희수: 스포츠과학과/기계공학부
* 김민수: 기계공학부
* 이중구: 기계공학부/전자전기공학부

## 알고리즘
### Perception
1. Vision 데이터 처리
	1. 차선 인식
	2. 표지판 인식
  
2. LiDAR 데이터 처리
	1. 장애물 인식
### Planning
3. 경로 설정
	1. 장애물과 차선 회피 경로 설정
	2. 미션 별 경로 계획
### Control
4. 제어
	1. 조향 제어
	2. 속도 제어
### Communication
5. 통신
	1. 전달할 통신 패킷 생성 및 플랫폼 전달
	2. 받은 통신 패킷 해석
### Operation
6. 연산 속도 상승
	1. CUDA for python
	2. 메인 프로세스 관리 및 주행 중 모니터링
