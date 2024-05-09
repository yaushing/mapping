###############
### IMPORTS ###
###############
from hub import port, motion_sensor
import motor_pair
import runloop
import math, time



#################
### VARIABLES ###
#################
scale = 360
coords = [0, 0]
bearing = 0



###############
### CLASSES ###
###############
class LinearRegression:
        def __init__(self):
            self.coef_ = []

        def fit(self, X, y):
            x = self._add_intercept(X)
            self.coef_ = self._normal_equation(x, y)
            return self

        def predict(self, X):
            predicted = []
            for i in range(len(X)):
                term = 0
                X[i] = X[i][::-1]
                for j in range(len(X[i])):
                    term += X[i][j] * self.coef_[j]
                term += self.coef_[-1]
                predicted.append(term)
            return predicted

        def _add_intercept(self, X):
            final = X[:][:]
            for i in range(len(final)):
                final[i].insert(0, 1)
            return final

        def _normal_equation(self, X, y):
            '''
            Normal equation:
            β = (XᵀX)⁻¹ x (XᵀY),
            Where X is the model matrix, with n+1 columns and N rows,
                n is the desired order of polynomial regression and
                N is the number of data points, which we fill as follows:
                The first column we fill with ones.
                The second with the observed values x₁, ..., xN of the independent variable.
                The third with x₁², ..., xN² of these values.
                The fourth with x₁³, ..., xN³,
                The n+1-th column with x₁ⁿ, ..., xNⁿ
            Y is a column matrix of the values y₁ to yN, of the values of the dependant values,
            β is a column matrix of the coefficients, from a₀ to aN,
                Such that the equation is the following:
                f(x) = a₀ + a₁x + a₂x² + a₃x³ + ...
            Xᵀ is the transpose of X,
            (XᵀX)⁻¹ is the inverse of XᵀX, and
            The operation between two matrices are matrix multiplication
            '''
            XT = transpose(X)
            XTX = matrix_multiplication(XT, X)
            XTXneg1 = scalar_matrix(transpose(cofact(XTX)), 1/det(XTX))
            XTXneg1XT = matrix_multiplication(XTXneg1, XT)
            coefficients = matrix_multiplication(XTXneg1XT, y)
            coefficients = [round(float(i[0]), 5) for i in coefficients][::-1]
            return coefficients

        def score(self, x, target_y):
            ysum = 0
            for y1 in target_y:
                ysum += y1[0]
            ymean = ysum / len(target_y)
            Sres = 0
            for i in range(len(target_y)):
                Sres += (target_y[i][0] - self.predict(poly([[i] for i in [x[i][0]]], len(self.coef_) - 1))[0])**2
            Stot = 0
            for i in range(len(target_y)):
                Stot += (target_y[i][0] - ymean)**2
            R2 = 1 - (Sres / Stot)
            return R2



#########################
### BACKEND FUNCTIONS ###
#########################
def poly(x, deg):
        final = []
        for i in x:
            x_li = []
            for j in range(1, deg+1):
                x_li.append(i[0] ** j)
            final.append(x_li)
        return final

def matrix_multiplication(matrix1, matrix2):
    if len(matrix1[0]) != len(matrix2):
        raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")
    rows1 = len(matrix1)
    cols1 = len(matrix1[0])
    cols2 = len(matrix2[0])
    result = [[0 for _ in range(cols2)] for _ in range(rows1)]
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result

def transpose(matrix):
    final_matrix = []
    temp_matrix = []
    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            temp_matrix.append(matrix[j][i])
        final_matrix.append(temp_matrix)
        temp_matrix = []
    return final_matrix

def det(matrix):
    if len(matrix) != len(matrix[0]): raise ValueError("Matrix is wrong.")
    if len(matrix) < 2: return matrix[0][0]
    if len(matrix) == 2: return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det_re = 0
    for col in range(len(matrix)):
        det_re += matrix[0][col] * det([i[:col] + i[col + 1:] for i in matrix[1:]]) * (-1) ** col
    return det_re

def cofact(matrix):
    final = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            resultant = [a[:j] + a[j + 1:] for a in (matrix[:i] + matrix[i + 1:])]
            final[i][j] = det(resultant)
            if (i % 2 == 0 and j % 2 != 0) or (i % 2 != 0 and j % 2 == 0): final[i][j] *= -1
    return final

def scalar_matrix(matrix, multiplicand):
    final = [[0 for _ in range(len(matrix))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            final[i][j] = matrix[i][j] * multiplicand
    return final

def regress(target_co_ords):
    co_ords = target_co_ords
    x = [[i[0]] for i in co_ords]
    y = [[i[1]] for i in co_ords]
    r_sq = 0
    degree = 0
    while r_sq < 0.99999:
        degree += 1
        x_ = poly(x, degree)
        model = LinearRegression().fit(x_, y)
        r_sq = model.score(x, y)
    coefficients = model.coef_ # Warning - ignore
    equation = ""
    for i in range(degree):
        equation += str(coefficients[i]) + "x^" + str(degree - i) + " + "
    equation += str(coefficients[-1])
    print("R\u00B2 = " + str(r_sq) + ". Degree = " + str(degree))
    print("Equation: " + equation + ".")
    return model # Warning - ignore

def f(x, coefficients):
    total = 0
    for i in range(len(coefficients)):
        total += coefficients[-1 * (i + 1)] * ((x) ** i)
    return total
def c(x, r, coords):
    c1 = coords[1] + (r**2 - (x - coords[0])**2)**0.5
    c2 = coords[1] - (r**2 - (x - coords[0])**2)**0.5
    return (c1, c2)

def solve(r, division, coords, coefficients, precision):
    intersections = []
    for i in range(int((coords[0] - r) * division), int((coords[0] + r) * division + 1)):
        #print("x = " + str(i/division) + " " + str(c(i/division, r, coords)) + " " + str(f(i/division, coefficients)))
        if (abs(c(i/division, r, coords)[0] - f(i / division, coefficients)) < precision) or (abs(c(i/division, r, coords)[1] - f(i / division, coefficients)) < precision):
            #print("(" + str(i/division) + "," + str(f(i/division, coefficients)) + ") is an intercept. Error of " + str(abs(c(i/division, r, coords)[0] - f(i / division, coefficients)))+ ")")
            #print("Saving: (" + str(round(i/division, 2)) + "," + str(round(f(i/division, coefficients), 2)) + ")")
            intersections.append([round(i/division, 2), round(f(i/division, coefficients), 2)])
    return intersections

def update_coords(coords, bearing):
    print("New Coords: " + str(coords) + " | Bearing: " + str(bearing))
    #| Angle: {math.degrees(math.atan(bearing))}

def arc(current_coords, target_coords, current_bearing, look_ahead, W):
    R = 0
    t = 2
    current_angle = round(math.degrees(math.atan(current_bearing)), 3)
    mAB = (target_coords[1] - current_coords[1]) / (target_coords[0] - current_coords[0])
    wanted_angle = round(math.degrees(math.atan(mAB)), 3)
    angle_error = round(abs(current_angle - wanted_angle), 3)
    R = (look_ahead / 2) / math.sin(math.radians(angle_error))
    inner = R - (W / 2)
    outer = R + (W / 2)
    innerArcLength = math.radians(2 * angle_error) * inner
    outerArcLength = math.radians(2 * angle_error) * outer
    outer = 1 if wanted_angle > current_angle else -1
    innerSpeed = int(innerArcLength * scale / t)
    outerSpeed = int(outerArcLength * scale / t)
    print(current_bearing, mAB, outer)
    print(current_angle, wanted_angle, angle_error, R)
    print(innerArcLength, outerArcLength)
    print(innerSpeed, outerSpeed)
    if outer == 1:
        motor_pair.move_tank_for_time(motor_pair.PAIR_1, 5000, 150, 406)

def check_bearing():
    return math.tan(motion_sensor.tilt_angles()[0] / 10)



##########################
### FRONTEND FUNCTIONS ###
##########################
async def curve(coords, bearing, target_co_ords, look_ahead, W = 10, pure_pursuit = True, path = False):
    model = regress(target_co_ords)
    direction = 1 if target_co_ords[-1][0] > coords[0] else -1
    if pure_pursuit:
        while ((coords[0] < target_co_ords[-1][0] and direction == 1) or (coords[0] > target_co_ords[-1][0] and direction == -1)):
            possible_targets = solve(look_ahead, 100, coords, model.coef_, 0.01)
            precision = 0.05
            while len(possible_targets) == 0 or possible_targets[-1][0] < coords[0]:
                if precision >= 1:
                    possible_targets = [[coords[0] + 1, f(coords[0] + 1, model.coef_)]]
                    break
                precision *= 5
                possible_targets = solve(look_ahead, 100, coords, model.coef_, precision)
            chosen_target = possible_targets[-1]
            print(coords, chosen_target)
            print(look_ahead, ((chosen_target[0] - coords[0])**2 + (chosen_target[1] - coords[1])**2)**0.5)
            follow_arc = arc(coords, chosen_target, bearing, look_ahead, W)
            coords = chosen_target
            coords = [round(coords[0], 2), round(coords[1], 2)]
            break
        return coords, bearing
    else:
        while ((coords[0] < target_co_ords[-1][0] and direction == 1) or (coords[0] > target_co_ords[-1][0] and direction == -1)):
            x = round(coords[0] + (direction * look_ahead), 2)
            y = round(f(x, model.coef_), 2)
            coords = [x, y]
        return coords, bearing

async def go_to(motorpair, target_co_ords, turn = True):
    direction = -1
    change = [target_co_ords[0] - coords[0], target_co_ords[1] - coords[1]]
    total_move = math.sqrt(change[0] ** 2 + change[1] ** 2)
    degrees_turn = round(math.degrees(math.atan(change[1]/change[0])), 2)
    if change[1] >= 0:
        if degrees_turn < 0: degrees_turn * -1 + 90
    else:
        direction = 1
        if degrees_turn < 0: degrees_turn * -1
        else: 180 - degrees_turn
    print(total_move * scale, degrees_turn, direction)
    await turn_degrees(motorpair, degrees_turn, direction)
    await motor_pair.move_for_degrees(motorpair, int(total_move * scale), 0)
    if turn: await turn_degrees(motorpair, degrees_turn, -1 * direction)

async def turn_degrees(motorpair, degrees, direction):
    offset = motion_sensor.tilt_angles()[0] / 10
    degrees_to_turn = degrees * direction
    while abs(motion_sensor.tilt_angles()[0] / 10 - offset) < abs(degrees_to_turn):
        motor_pair.move_tank(motorpair, 180 * direction, -180 * direction)
    motor_pair.stop(motorpair)
        #print(math.atan(check_bearing()))
    #motor_pair.move_tank_for_degrees(motorpair, -1*degrees*4, degrees*4, 1000)



############
### MAIN ###
############
async def main():
    motor_pair.pair(motor_pair.PAIR_1, port.A, port.B)
    print(check_bearing())
    #await go_to(motor_pair.PAIR_1, [2, 1.5])
    #await curve([0, 0], 0, [[0, 0], [2, 2], [4, 0]], 0.5)
    arc([0, 0], [1, 1], 0, 1.414, 0.875)
    time.sleep(5)
    print(check_bearing())
runloop.run(main())
