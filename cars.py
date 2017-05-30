import math

FRAME_COUNT_THRESHOLD = 6

def get_car_center(car):
    x1 = car[0][0]
    y1 = car[0][1]
    x2 = car[1][0]
    y2 = car[1][1]
    return (int((x1+x2)/2), int((y1+y2)/2))

def check_is_same_car(car1, car2):
    car1_center = car1[1]
    car2_center = car2[1]
    car1_x = car1_center[0]
    car1_y = car1_center[1]
    car2_x = car2_center[0]
    car2_y = car2_center[1]
    delta = 64/2  # Half of smallest window size that we check for cars

    if (abs(car1_x - car2_x) < delta) and (abs(car1_y - car2_y) < delta):
        return True

    car1_window = car1[0]
    car2_window = car2[0]    

    # check for overlap
    if (car1_window[0][0] <= car2_window[0][0]) and (car1_window[1][0] >= car2_window[1][0]) and (car1_window[0][1] <= car2_window[0][1]) and (car1_window[1][1] >= car2_window[1][1]):
        # car 1 superset of car 2
        return True

    if (car2_window[0][0] <= car1_window[0][0]) and (car2_window[1][0] >= car1_window[1][0]) and (car2_window[0][1] <= car1_window[0][1]) and (car2_window[1][1] >= car1_window[1][1]):
        # car 2 superset of car 1
        return True

    return False


class Cars:
    def __init__(self):
        self.car_list = []
        self.cars_tracked = []

    def match_cars_with_tracked(self, new_car):
        match = False
        car = None
        for known_car in self.cars_tracked:
            if check_is_same_car(new_car, known_car):
                known_car[4] = 1
                print("Found match")
                car = known_car
                match = True
                break;
        return match, car
        
    def updateCarsTracked(self, car_list):
        self.car_list = car_list
        if len(self.cars_tracked) == 0:
            if len(self.car_list) > 0:
                print("Adding {} cars to empty tracked list".format(len(self.car_list)))
                self.cars_tracked = self.car_list
        else:
            for car in self.cars_tracked:
                # mark nothing is tracked
                car[4] = 0
            for car in self.cars_tracked:
                # mark nothing is tracked
                if car[4]:
                    print("What happened !!!!!!!!!")
            new_cars_tracked = []
            # Match the cars in new frame with tracked cars
            for car in car_list:
                match, known_car = self.match_cars_with_tracked(car)
                if match:
                    car[4] = 1
                    car[3] = 0
                    car[2] = known_car[2] + 1
                    print("Update match")
                    new_cars_tracked.append(car.copy())
                else:
                    print("Add new car")
                    new_cars_tracked.append(car.copy())
                    
            # go over tracked cars that were not matched and see to include them or retire them.
            for known_car in self.cars_tracked:
                if not known_car[4]:
                    if (known_car[3] >= FRAME_COUNT_THRESHOLD) or (known_car[3] > known_car[2]):
                        print("Retiring car")
                    else:
                        print("Add previously tracked car")
                        known_car[3] += 1
                        new_cars_tracked.append(known_car.copy())
            self.cars_tracked = new_cars_tracked
        
        # return cars windows to be drawn
        cars_to_draw = []
        for car in self.cars_tracked:
            cars_to_draw.append(car[0])
        print("Returning {} cars to draw".format(len(cars_to_draw)))

        return cars_to_draw

    def updateNewCars(self, cars):
        car_list = []
        for car in cars:
            center = get_car_center(car)
            # format is window, center, hits, miss, tracked in current round
            car_list.append([car, center, 1, 0, 0])
        cars_to_draw = self.updateCarsTracked(car_list)
        return cars_to_draw
