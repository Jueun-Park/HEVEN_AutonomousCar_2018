class Parabola:
    # y = a0 + a1 * x + a2 * x^2
    def __init__(self, a0, a1, a2):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2

    # 곡률 구하는 함수
    def get_curvature(self, x):
        curvature = 2 * self.a2 / pow((1 + (self.a1 + 2 * self.a2 * x) ** 2), 1.5)
        return curvature

    # 기울기 구하는 함수
    def get_derivative(self, x):
        derivative = self.a1 + 2 * self.a2 * x
        return derivative

    # 함숫값 구하는 함수
    def get_value(self, x):
        value = self.a0 + self.a1 * x + self.a2 * x ** 2
        return value
