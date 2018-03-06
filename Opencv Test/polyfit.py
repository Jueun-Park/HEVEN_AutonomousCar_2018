from numpy.polynomial.polynomial import polyfit  # polyfit: 추세선 계수 구하는 함수

xs = []  # 점들의 x좌표를 담는 list
ys = []  # 점들의 y좌표를 담는 list

f = open('C:/Users/jglee/Documents/카카오톡 받은 파일/data.txt', 'r')  # 모든 점들의 좌표가 담긴 data.txt 열기

total_points = f.read().split('\n')

for point in total_points:
    xs.append(float(point.split(',')[0]))
    ys.append(float(point.split(',')[1]))

f.close()

p2 = polyfit(xs, ys, 2)

print(p2)