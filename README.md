# HEVEN_AutonomousCar_2018
2018 International Student Car Competition: Autonomous Car SKKU Team. HEVEN

[위키 작성법(마크다운 문서)](https://gist.github.com/ihoneymon/652be052a0727ad59601)

## 팀원
김성우 김홍빈 박주은 박준혁 유성룡

김민수 이중구 이용호

예비 팀원: 김윤진 김진웅 예하진 유재현 이민영 이아영 현지웅

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
  
## 비전 데이터 파싱 및 처리
### 팀원
#### 차선
* 김홍빈
#### 표지판
* 김성우
### 담당
* 차선 인식
* 표지판 인식
* 비전 머신러닝 (GPU 이용)
### 프로그램 명
* `lane_cam.py`: 차선 인식
* `sign_cam.py`: 표지판 인식

## 라이다 데이터 파싱 및 처리
### 팀원
* 김성우
* 김홍빈
### 담당
* 라이다 통신, 장애물 인식
### 프로그램 명
* `lidar.py`

## 주행 알고리즘
### 팀원
* 김홍빈
* 박준혁
### 담당
* 경로 탐색 알고리즘
* 제어 알고리즘
* 미션 별 알고리즘: `주차, 유턴, 동적 장애물, 횡단보도 등`
### 프로그램 명
* `path_planner.py`
* `car_control.py`

## CPU 분산처리
### 팀원
* 유성룡
### 담당
* 연산 속도 상승
* 프로그램 병렬 처리
* CUDA for python
### 프로그램 명
* `main.py`
* 연산량이 많은 프로그램

## 통신
### 팀원
* 박주은
### 담당
* 플랫폼-데스크탑 패킷 통신
### 프로그램 명
* `communication.py`
