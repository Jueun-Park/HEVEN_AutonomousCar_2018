# 차량 제어 (기본, 미션별)
# input: 1. 경로가 표시된 numpy array (from path_planner)
#        2. 인식한 유턴_표지판_위치_리스트 정보 (from sign_cam)
#        3. 통신 패킷 정보, 형식 미정 (from communication)
# output: 통신 패킷 만드는 데 필요한 정보 (to communication)

import numpy
