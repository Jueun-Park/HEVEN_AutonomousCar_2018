import cv2
import numpy as np


def findCenterofMass(src):
    sum_of_y_mass_coordinates = 0
    num_of_mass_points = 0

    for y in range(0, len(src)):
        for x in range(0, len(src[0])):
            if src[y][x] == 255:
                sum_of_y_mass_coordinates += y
                num_of_mass_points += 1

    if num_of_mass_points == 0:
        center_of_mass_y = int(round(len(src) / 2))

    else:
        center_of_mass_y = int(round(sum_of_y_mass_coordinates / num_of_mass_points))

    return center_of_mass_y

def vision() :

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 180, 180])

    lower_grey = np.array([150, 150, 150])
    upper_grey = np.array([210, 210, 210])

    video_left = cv2.VideoCapture('C:/Users/Administrator/Desktop/left.avi')
    video_right = cv2.VideoCapture('C:/Users/Administrator/Desktop/right.avi')

    previous_points = None
    current_points = np.array([0] * 10)

    coefficients = None

    while True:

        ret_left, frame_left = video_left.read()
        ret_right, frame_right = video_right.read()

        left_cropped = frame_left[210:510, 262:562]
        right_cropped = frame_right[151:451, 0:300]

        left_transposed = cv2.flip(cv2.transpose(left_cropped), 0)
        right_transposed = cv2.flip(cv2.transpose(right_cropped), 0)

        left_black_filtered = cv2.inRange(left_transposed, lower_black, upper_black)
        right_black_filtered = cv2.inRange(right_transposed, lower_black, upper_black)

        left_grey_filtered = cv2.inRange(left_transposed, lower_grey, upper_grey)
        right_grey_filtered = cv2.inRange(right_transposed, lower_grey, upper_grey)

        left_filtered = cv2.bitwise_not(left_black_filtered + left_grey_filtered)
        right_filtered = cv2.bitwise_not(right_black_filtered + right_grey_filtered)

        if previous_points is None:
            row_sum = np.sum(right_filtered[0:300, 270:300], axis=1)[100:200]
            start_point = np.argmax(row_sum) + 100
            current_points[0] = start_point

            for i in range(1, 10):
                reference = current_points[i - 1] - 20
                small_rectangle = right_filtered[current_points[i - 1] - 20:current_points[i - 1] + 20,
                                  300 - 30 * i:330 - 30 * i]

                current_points[i] = reference + findCenterofMass(small_rectangle)

        else:
            for i in range(0, 10):
                current_points[i] = previous_points[i] - 20 + findCenterofMass(
                    right_filtered[previous_points[i] - 20:previous_points[i] + 20, 270 - 30 * i:300 - 30 * i])

        previous_points = current_points

        xs = np.array([30 * i for i in range(10, 0, -1)]) - 300
        ys = 300 - current_points

        coefficients = np.polyfit(xs, ys, 2)
        print(coefficients)

        xs_plot = np.array([1 * i for i in range(301, 0, -1)]) - 300
        ys_plot = np.array([coefficients[2] + coefficients[1] * v + coefficients[0] * v ** 2 for v in xs_plot])

        transformed_x = xs_plot + 300
        transformed_y = 300 - ys_plot

        for i in range(0, 301):
            cv2.circle(right_transposed, (int(transformed_x[i]), int(transformed_y[i])), 2, (0, 0, 200), -1)

        # for i in range(0, 10):
        # cv2.line(right_filtered, (300 - 30 * i, current_points[i] - 20), (300 - 30 * i, current_points[i] + 20), 140)

        cv2.imshow('right', right_transposed)
        cv2.imshow('left', left_transposed)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    video_left.release()
    video_right.release()
    cv2.destroyAllWindows()

