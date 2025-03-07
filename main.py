import numpy as np
import pygame
from typing import Self
import sys
import pickle
import copy
from multiprocessing import Process


#誘因粒子：AttractantParticle:AP

rng = np.random.default_rng()

FOOD_RADIUS = np.int16(5)
CELL_RADIUS = np.int16(10)
FOOD_ENERGY = np.float32(5)
SHOOTING_AP_RATE = np.float32(0.7)
AP_LIFETIME = np.int16(300)
AP_SPEED = np.float32(10)
CELL_DUPLICATE_RATE = np.float32(0.5)
DUPLICATE_COOL_DOWN = np.int16(900)
RAY_COLOR = np.array((1,1,0),dtype=np.float32)
AP_COLOR = np.array((0.5,1,1),dtype=np.float32)
NEURAL_NETWORK_MUTATE_RATE = np.float32(0.01)
ABSORPTION_RATE = np.float32(1)
RAYS_COUNT = 32
RAY_RANGE = 50
TOLERANCE = 10

class Object:
    def __init__(self, energy:np.float32, colors:np.array(np.float32), position:np.array(np.float32), radius:np.int16, game:"Game"):
        self.energy = np.float32(energy)
        self.colors = np.array(colors) #(R, G, B)それぞれ0~1
        self.position = np.array(position)
        self.radius = np.int16(radius)
        self.pre_movement = np.array([0.0, 0.0])
        self.game = game
        self.remove_flag = False

    def __str__(self):
        return "Object"

    def get_remove_flag(self):
        return self.remove_flag

    def get_info(self, info_type:str):
        """
        :return: energy, colors, position, radius, pre_movement, remove_flag
        """
        if info_type == "energy":
            return self.energy
        elif info_type == "colors":
            return self.colors[:]
        elif info_type == "position":
            return self.position[:]
        elif info_type == "radius":
            return self.radius
        elif info_type == "pre_movement":
            return self.pre_movement[:]
        elif info_type == "remove_flag":
            return self.remove_flag
        else:
            raise ValueError("情報取得時のタイプ指定が間違っています。L37")

    def true_remove_flag(self):
        self.remove_flag = True

    def move(self, vector:np.array(np.float32)):
        """
        vector分動く
        :param vector:
        :return:
        """
        world_size = self.game.get_world_size()
        self.position += vector
        self.position = np.array((self.position[0] % world_size[0], self.position[1] % world_size[1]))

    def premove_to_move(self):
        self.move(self.pre_movement)
        self.pre_movement = np.zeros(2,dtype=np.float32)

    def no_energy_check(self):
        """
        エネルギーが0になったらremove_flagをTrueにする
        :return:
        """
        if self.energy <= 0:
            self.true_remove_flag()

    def add_energy_and_return_added_energy(self, value:np.float32):
        """
        エネルギー値を変え、エネルギー変更量(符号そのまま)を返す
        :param value:
        :return:
        """
        if self.energy <= 0:
            self.energy = 0
            return np.float32(0.0)
        else:
            self.energy += value
            return value

    def transfer_energy(self, target:Self, value:np.float32):
        """
        エネルギーを引き渡す(自分から相手へ)
        :param target:
        :param value:
        :return:
        """
        transferring_energy = self.add_energy_and_return_added_energy(-value)
        target.add_energy_and_return_added_energy(-transferring_energy)

    def AP_receptor(self, AP):
        """
        誘因粒子が当たった際に関数が呼ばれ、オブジェクトを動かす。
        :param AP:
        :return:
        """
        velocity_vector = AP.get_info("velocity_vector")
        self.pre_movement -= 5 * velocity_vector
        AP.true_remove_flag()


class Food(Object):
    def __init__(self, energy:np.float32, colors:np.array(np.float32), position:np.array(np.float32), radius:np.int16, game):
        super().__init__(energy, colors, position, radius, game)

    def __str__(self):
        return "Food"





class Cell(Object):
    def __init__(self, energy:np.float32, colors:np.array(np.float32), position:np.array(np.float32), radius:np.int16, timers_max_cycles:np.array(np.int16), game:"Game"):
        super().__init__(energy, colors, position, radius, game)
        self.timers_max_cycles = np.array(timers_max_cycles)
        self.timers_cycle = [0, 0, 0, 0, 0]
        self.timer_duplicate = 0
        self.energy_absorption_rate = ABSORPTION_RATE
        self.tickly_energy_consumption = np.float32(0.5)
        self.weight_input_to_hidden = np.zeros((30,40),dtype=np.float32)
        self.bias_input_to_hidden = np.zeros((30,1),dtype=np.float32)
        self.weight_hidden_to_output = np.zeros((21,30),dtype=np.float32)
        self.bias_hidden_to_output = np.zeros((21,1),dtype=np.float32)
        self.neural_network_outputs = np.zeros((21, 1),dtype=np.float32)
        self.delta_position = np.array([0.0, 0.0])
        self.age = 0

    cell_max_age = 1000

    def __str__(self):
        return "Cell"

    def set_neural_network(self, weight_input_to_hidden, bias_input_to_hidden, weight_hidden_to_output, bias_hidden_to_output):
        self.weight_input_to_hidden = weight_input_to_hidden
        self.bias_input_to_hidden = bias_input_to_hidden
        self.weight_hidden_to_output = weight_hidden_to_output
        self.bias_hidden_to_output = bias_hidden_to_output

    def set_neural_network_random(self):
        self.weight_input_to_hidden = 0.5 * rng.random(self.weight_input_to_hidden.shape) - 0.25
        self.bias_input_to_hidden = 0.2 * rng.random(self.bias_input_to_hidden.shape) - 0.1
        self.weight_hidden_to_output = 0.5 * rng.random(self.weight_hidden_to_output.shape) - 0.25
        self.bias_hidden_to_output = 0.2 * rng.random(self.bias_hidden_to_output.shape) - 0.1

    def premove(self, vector:np.array(np.float32)): #外部からの処理のためのもの
        self.pre_movement = vector

    def premove_to_move(self):
        self.move(self.pre_movement)
        self.delta_position = self.pre_movement[:]
        self.pre_movement = np.zeros(2,dtype=np.float32)

    def set_duplicate_time(self,duplicate_time):
        self.timer_duplicate = duplicate_time % DUPLICATE_COOL_DOWN

    def add_age(self):
        self.age += 1

    def check_age(self):
        if self.age >= self.cell_max_age:
            self.true_remove_flag()

    def absorb_energy_from_object(self, objectA):
        if touch_judgement_between_objects(self, objectA):
            self.transfer_energy(objectA, -self.energy_absorption_rate)

    def cycly_consume_energy(self):
        self.energy -= self.tickly_energy_consumption

    def update_timers(self):
        for k, timer in enumerate(self.timers_cycle):
            self.timers_cycle[k] = np.mod(self.timers_cycle[k]+1, self.timers_max_cycles[k])
        if self.timer_duplicate != 0:
            self.timer_duplicate = np.mod(self.timer_duplicate + 1, DUPLICATE_COOL_DOWN) #+1は増殖時に。

    def info_collector_for_NN(self):
        """
        ニューラルネットワークへ渡す情報を集める
        :return:
        """
        rays_info = self.shoot_rays()
        self_energy = self.energy
        self_delta_position = self.delta_position[:]
        timers_info = self.timers_cycle
        info_for_NN = np.concatenate((rays_info, np.array([self_energy, *self_delta_position]), timers_info), axis=0)
        info_for_NN = info_for_NN.reshape((info_for_NN.size, 1))
        return info_for_NN[:]

    def calc_neural_network(self, NN_input):
        input_layer:np.array(np.float32) = np.array(NN_input).reshape((40,1))
        hidden_layer:np.array(np.float32) = np.tanh((self.weight_input_to_hidden @ input_layer + self.bias_input_to_hidden) * 0.2)
        output_layer:np.array(np.float32) = np.tanh((self.weight_hidden_to_output @ hidden_layer + self.bias_hidden_to_output) * 0.2)
        self.neural_network_outputs = output_layer.reshape(output_layer.size)

    def shoot_rays(self):
        thetas = np.linspace(0, 2 * np.pi, RAYS_COUNT, endpoint=False)
        ans_bools = [self.game.shoot_ray(self.position, theta) for theta in thetas]
        return [np.float32(1) if ans_bool is True else np.float32(-1) for ans_bool in ans_bools]

    def shoot_AP(self):
        neural_network_outputs_AP = self.neural_network_outputs[0:2]
        AP_shoot_flag = neural_network_outputs_AP[0]
        if AP_shoot_flag > 1 - SHOOTING_AP_RATE:
            AP_shoot_theta = neural_network_outputs_AP[1] * np.pi
            AP_vector = np.array([np.cos(AP_shoot_theta), np.sin(AP_shoot_theta)])
            return AttractantParticle(self.position + AP_vector * (self.radius + 1), AP_vector * AP_SPEED, AP_LIFETIME, self.game)
        else:
            return None

    def duplicate(self):
        neural_network_outputs_duplicate = self.neural_network_outputs[2:5]
        duplicate_flag = neural_network_outputs_duplicate[0]
        duplicate_energy_ratio = neural_network_outputs_duplicate[1:3]
        duplicate_energy_ratio = (duplicate_energy_ratio + 1) * 0.9 / np.sum(duplicate_energy_ratio + 1 + 1e-10)
        if duplicate_flag > 1 - CELL_DUPLICATE_RATE * 2:
            if self.timer_duplicate == 0:
                cell1_energy = self.energy * duplicate_energy_ratio[0]
                cell2_energy = self.energy * duplicate_energy_ratio[1]
                random_position = self.position + 0.5 * rng.random(2)
                random_position = np.mod(random_position, self.game.world_size)
                self.energy = cell1_energy
                cell2 = Cell(cell2_energy, self.colors[:], random_position, self.radius, self.timers_max_cycles, self.game)
                mutated_weight_input_to_hidden = mutate_neural_network(self.weight_input_to_hidden)
                mutated_bias_input_to_hidden = mutate_neural_network(self.bias_input_to_hidden)
                mutated_weight_hidden_to_output = mutate_neural_network(self.weight_hidden_to_output)
                mutated_bias_hidden_to_output = mutate_neural_network(self.bias_hidden_to_output)
                cell2.set_neural_network(mutated_weight_input_to_hidden, mutated_bias_input_to_hidden, mutated_weight_hidden_to_output, mutated_bias_hidden_to_output)
                self.timer_duplicate += 1
                cell2.set_duplicate_time(self.timer_duplicate)
                return cell2
        else:
            return None

    def move_with_neural_network_output(self):
        neural_network_outputs_move = self.neural_network_outputs[5:21].reshape(16)
        thetas = np.linspace(0,np.pi,16,endpoint=False)
        vectors = [np.array((np.cos(theta), np.sin(theta))) for theta in thetas]
        for vector, k in zip(vectors, neural_network_outputs_move):
            self.pre_movement += k * vector

    def get_neural_network(self):
        return self.weight_input_to_hidden, self.bias_input_to_hidden, self.weight_hidden_to_output, self.bias_hidden_to_output



class Particle:
    def __init__(self, position:np.array(np.float32), velocity_vector:np.array(np.float32), lifetime:np.int16, game:"Game"):
        self.position = position
        self.velocity_vector = velocity_vector
        self.lifetime = lifetime
        self.age = 0
        self.game = game
        self.remove_flag = False

    energy_consumption = np.float32(1.0)
    particle_speed = np.float32(10.0)

    def get_remove_flag(self):
        return self.remove_flag

    def get_info(self, info_type:str):
        """
        :return: energy, position, velocity_vector, lifetime, age, self.remove_flag
        """
        if info_type == "position":
            return self.position[:]
        elif info_type == "velocity_vector":
            return self.velocity_vector[:]
        elif info_type == "lifetime":
            return self.lifetime
        elif info_type == "age":
            return self.age
        elif info_type == "remove_flag":
            return self.remove_flag
        else:
            raise ValueError("情報取得時のタイプ指定が間違っています。L182")

    def true_remove_flag(self):
        self.remove_flag = True

    def add_age(self):
        self.age += 1

    def age_check(self):
        if self.age >= self.lifetime:
            self.remove_flag = True

    def move(self):
        """
        決まった方向(velocity_vector)に動く
        :return:
        """
        world_size = self.game.get_world_size()
        self.position += self.velocity_vector
        self.position = np.array((self.position[0] % world_size[0], self.position[1] % world_size[1]))

    def lifetime_check(self):
        """
        粒子の寿命チェック。寿命を過ぎたらremove_flagをTrueにする
        :return:
        """
        if self.lifetime <= self.age:
            self.remove_flag = True


class AttractantParticle(Particle):
    def __init__(self, position:np.array(np.float32), velocity_vector:np.array(np.float32), lifetime:np.int16, game:"Game"):
        super().__init__(position, velocity_vector, lifetime, game)



def touch_judgement_between_particle_and_object(particle:Particle, objectA:Object):
    particle_position = particle.get_info("position")
    objectA_position = objectA.get_info("position")
    objectA_radius = objectA.get_info("radius")
    if np.linalg.norm(particle_position - objectA_position) <= objectA_radius:
        return True
    else:
        return False


def touch_judgement_between_objects(objectA:Object, objectB:Object):
    distance = np.linalg.norm(objectA.get_info("position") - objectB.get_info("position"))
    if distance <= objectA.get_info("radius") + objectB.get_info("radius"):
        return True
    else:
        return False

def calc_repulsion_between_cells(main_cell:Cell, sub_cell:Cell): #MainCellが動く方
    if main_cell != sub_cell:
        peak_height = 1.0
        standard_deviation = 2
        sub_to_main = main_cell.get_info("position") - sub_cell.get_info("position")
        distance = np.linalg.norm(sub_to_main)
        repulsion = peak_height * np.exp(-distance**2 / standard_deviation**2)
        return repulsion
    else:
        return np.array((0.0, 0.0))

def mutate_neural_network(matrix, mutate_rate=NEURAL_NETWORK_MUTATE_RATE):
    for vector in matrix:
        for arg_number in range(len(vector)):
            if rng.random() < NEURAL_NETWORK_MUTATE_RATE:
                vector[arg_number] = rng.random() * 2 - 1
    return matrix

def mutate_all_neural_network(neural_networks: list, mutate_rate=NEURAL_NETWORK_MUTATE_RATE):
    weight_input_to_hidden, bias_input_to_hidden, weight_hidden_to_output, bias_hidden_to_output = copy.copy(neural_networks)
    weight_input_to_hidden = mutate_neural_network(weight_input_to_hidden[:], mutate_rate)
    bias_input_to_hidden = mutate_neural_network(bias_input_to_hidden[:], mutate_rate)
    weight_hidden_to_output = mutate_neural_network(weight_hidden_to_output[:], mutate_rate)
    bias_hidden_to_output = mutate_neural_network(bias_hidden_to_output[:], mutate_rate)
    return weight_input_to_hidden, bias_input_to_hidden, weight_hidden_to_output, bias_hidden_to_output


class Game:
    def __init__(self):
        pygame.init()
        self.screen_size = (600, 600)
        self.screen = pygame.display.set_mode(np.array(self.screen_size))
        pygame.display.set_caption("ArtificialLife")
        self.world_size = np.array((600.0, 600.0), dtype=np.float32)#原点は左下
        self.clock = pygame.time.Clock()
        self.running = True
        self.food_list:list[Food] = []
        self.cell_list:list[Cell] = []
        self.AP_list:list[AttractantParticle] = []
        self.energy_pool = np.float32(0.0)
        self.energy_per_food = FOOD_ENERGY

    def append_cells(self,cells:list[Cell]):
        for cell in cells:
            self.cell_list.append(cell)

    def append_foods(self,foods:list[Food]):
        for food in foods:
            self.food_list.append(food)

    def debug_append_APs(self,APs:list[AttractantParticle]):
        for ap in APs:
            self.AP_list.append(ap)

    def set_energy_pool(self, energy):
        self.energy_pool = energy

    def debug_randomize_cell_neural_network(self):
        for cell in self.cell_list:
            cell.set_neural_network_random()

    def get_world_size(self):
        return self.world_size

    def world_coordinate_to_screen_coordinate(self, coordinate):
        return self.world_size - coordinate

    def fill_screen_white(self):
        self.screen.fill((0,0,0))

    def get_cell_count(self):
        return len(self.cell_list)

    def shoot_ray(self, shooting_position, theta):
        # food_list から位置情報を NumPy 配列として取得 (food_listの構造に依存)
        food_positions = np.array([food.get_info("position") for food in self.food_list])

        # 相対位置ベクトルを計算
        relative_positions = food_positions - shooting_position

        # 各foodとの距離を計算
        distances = np.linalg.norm(relative_positions, axis=1)

        # 距離が0より大きく、RAY_RANGE以下のfoodのインデックスを取得
        valid_indices = np.where((distances > 0) & (distances <= RAY_RANGE))[0]

        if valid_indices.size > 0:  # 有効なfoodがある場合
            # 有効なfoodに対する正規化された相対位置ベクトルを計算
            normalized_relative_positions = relative_positions[valid_indices] / distances[valid_indices, None]

            # 射撃ベクトルを計算
            shoot_vector = np.array((np.cos(theta), np.sin(theta)))

            # 内積を計算
            dot_products = np.dot(normalized_relative_positions, shoot_vector)

            # 内積が1に近いfoodがあるか判定
            if np.any(np.abs(dot_products - 1) < TOLERANCE):
                return True

        return False


    def generate_food(self): ###1
        food_count = int(np.floor(self.energy_pool / self.energy_per_food))
        for _ in range(food_count):
            food_position = rng.random(2) * self.world_size
            self.food_list.append(Food(self.energy_per_food, rng.random((3,3)), food_position, FOOD_RADIUS, self))
            self.energy_pool -= self.energy_per_food

    def APs_touch_judgement(self): ###2 #AI
        if not self.AP_list or not self.cell_list and not self.food_list: # APリスト、細胞リスト、食物リストのいずれかが空の場合は処理不要
            return

        ap_positions = np.array([ap.position for ap in self.AP_list])
        cell_positions = np.array([cell.position for cell in self.cell_list])
        food_positions = np.array([food.position for food in self.food_list])

        for ap_index, ap in enumerate(self.AP_list):
            # Cellとの接触判定
            for cell_index, cell in enumerate(self.cell_list):
                distance = np.linalg.norm(ap_positions[ap_index] - cell_positions[cell_index])
                if distance <= cell.radius:
                    cell.AP_receptor(ap)

            # Foodとの接触判定
            for food_index, food in enumerate(self.food_list):
                distance = np.linalg.norm(ap_positions[ap_index] - food_positions[food_index])
                if distance <= food.radius:
                    food.AP_receptor(ap)

    def calc_repulsion(self): ###3 repulsion:斥力 #AI
        if not self.cell_list: # 細胞リストが空の場合は処理不要
            return

        cell_positions = np.array([cell.position for cell in self.cell_list]) # 細胞の位置をNumPy配列に
        n_cells = len(self.cell_list)
        repulsion_vectors = np.zeros((n_cells, 2), dtype=np.float32) # 反発力ベクトルを初期化

        for i in range(n_cells):
            for j in range(n_cells):
                if i != j:
                    main_cell = self.cell_list[i]
                    sub_cell = self.cell_list[j]
                    peak_height = 5.0
                    standard_deviation = 2
                    sub_to_main_vec = main_cell.position - sub_cell.position
                    distance_sq = np.sum(sub_to_main_vec**2) # 距離の二乗を計算 (平方根計算を避ける)
                    repulsion_magnitude = peak_height * np.exp(-distance_sq / standard_deviation**2)
                    repulsion_vec = repulsion_magnitude * sub_to_main_vec / (np.sqrt(distance_sq) + 1e-10) # 正規化と微小値加算
                    repulsion_vectors[i] += repulsion_vec

        for i in range(n_cells):
            self.cell_list[i].premove(repulsion_vectors[i])

    def energy_absorbing(self): ###4 #AI
        if not self.cell_list or not self.food_list and not self.cell_list: # 細胞リスト、食物リスト、細胞リストのいずれかが空の場合は処理不要
            return

        cell_positions = np.array([cell.position for cell in self.cell_list])
        food_positions = np.array([food.position for food in self.food_list])
        cell_radii = np.array([cell.radius for cell in self.cell_list])
        food_radii = np.array([food.radius for food in self.food_list])

        for i, cell in enumerate(self.cell_list):
            # Cell 同士のエネルギー吸収
            for j, sub_cell in enumerate(self.cell_list):
                if i != j:
                    distance = np.linalg.norm(cell_positions[i] - cell_positions[j])
                    if distance <= cell_radii[i] + cell_radii[j]:
                        cell.absorb_energy_from_object(sub_cell)
            # Food からのエネルギー吸収
            for j, food in enumerate(self.food_list):
                distance = np.linalg.norm(cell_positions[i] - food_positions[j])
                if distance <= cell_radii[i] + food_radii[j]:
                    cell.absorb_energy_from_object(food)

    def consume_cell_energy(self): ###5
        for cell in self.cell_list:
            cell.cycly_consume_energy()

    def update_cell_timers(self): ###6
        for cell in self.cell_list:
            cell.update_timers()

    def execute_neural_network(self): ###7
        for cell in self.cell_list:
            collected_info = cell.info_collector_for_NN()
            cell.calc_neural_network(collected_info)

    def shoot_APs(self): ###9
        for cell in self.cell_list:
            AP = cell.shoot_AP()
            if AP is not None:
                self.AP_list.append(AP)

    def cells_duplicate(self): ###10
        cell_list = self.cell_list[:]
        for cell in self.cell_list:
            cell2 = cell.duplicate()
            if cell2 is not None:
                cell_list.append(cell2)
        self.cell_list = cell_list

    def cells_move(self): ###11
        for cell in self.cell_list:
            cell.move_with_neural_network_output()

    def add_age_to_particle_and_cell(self): ###12
        for cell in self.cell_list:
            cell.add_age()
        for AP in self.AP_list:
            AP.add_age()

    def remove_flag_check(self): ###13
        for cell in self.cell_list:
            cell.no_energy_check()
            #cell.check_age()
        for food in self.food_list:
            food.no_energy_check()
        for AP in self.AP_list:
            AP.age_check()

    def premove_to_move(self): ###14
        for cell in self.cell_list:
            cell.premove_to_move()
        for food in self.food_list:
            food.premove_to_move()

    def move_particles(self): ###15
        for ap in self.AP_list:
            ap.move()

    def collect_statistics_data(self): ###16
        pass

    def remove_remove_flagged(self): ###17 #AI
        self.cell_list = [cell for cell in self.cell_list if not cell.get_remove_flag()]
        self.food_list = [food for food in self.food_list if not food.get_remove_flag()]
        self.AP_list = [AP for AP in self.AP_list if not AP.get_remove_flag()]

    def add_energy_to_energy_pool(self): ###18
        if len(self.food_list) < FOOD_ENERGY * 50:
            self.energy_pool += FOOD_ENERGY * 10

    def draw_cells(self):
        for cell in self.cell_list:
            screen_position = self.world_coordinate_to_screen_coordinate(cell.get_info("position") * self.screen_size[0] / self.world_size)
            color1, color2, color3 = np.array(cell.get_info("colors")*255, dtype=np.int16)
            radius1 = cell.get_info("radius")
            radius2 = radius1 - 1
            radius3 = radius2 - 1
            pygame.draw.circle(self.screen, color1, screen_position, radius1)
            pygame.draw.circle(self.screen, color2, screen_position, radius2)
            pygame.draw.circle(self.screen, color3, screen_position, radius3)

    def draw_foods(self):
        for food in self.food_list:
            screen_position = self.world_coordinate_to_screen_coordinate(food.get_info("position") * self.screen_size[0] / self.world_size)
            color1, color2, color3 = np.array(food.get_info("colors") * 255, dtype=np.int16)
            radius1 = food.get_info("radius")
            radius2 = radius1 - 1
            radius3 = radius2 - 1
            pygame.draw.circle(self.screen, color1, screen_position, radius1)
            pygame.draw.circle(self.screen, color2, screen_position, radius2)
            pygame.draw.circle(self.screen, color3, screen_position, radius3)

    def draw_rays(self):
        thetas = np.linspace(0, 2 * np.pi, RAYS_COUNT, endpoint=False)
        for cell in self.cell_list:
            ray_vectors = [np.array((np.cos(theta),np.sin(theta))) for theta in thetas]
            rays_start_point = [self.world_coordinate_to_screen_coordinate(cell.get_info("position") + ray_vector * (cell.get_info("radius") + 1)) for ray_vector in ray_vectors]
            rays_end_point = [self.world_coordinate_to_screen_coordinate(cell.get_info("position") + ray_vector * RAY_RANGE) for ray_vector in ray_vectors]
            for ray_start_point, ray_end_point in zip(rays_start_point, rays_end_point):
                pygame.draw.line(self.screen, (255, 0, 0), ray_start_point, ray_end_point, 1)

    def draw_APs(self):
        for AP in self.AP_list:
            screen_position = self.world_coordinate_to_screen_coordinate(AP.get_info("position") * self.screen_size[0] / self.world_size)
            color = AP_COLOR * 255
            radius = 2
            pygame.draw.circle(self.screen, color, screen_position, radius)



    def cycle(self, debug_mode = False):
        self.generate_food()
        self.APs_touch_judgement()
        self.calc_repulsion()
        self.energy_absorbing()
        self.consume_cell_energy()
        self.update_cell_timers()
        if debug_mode:
            self.debug_randomize_cell_neural_network()
        self.execute_neural_network()
        self.shoot_APs()
        self.cells_duplicate()
        self.cells_move()
        self.add_age_to_particle_and_cell()
        self.premove_to_move()
        self.move_particles()
        self.remove_flag_check()
        self.remove_remove_flagged()
        self.add_energy_to_energy_pool()





def main(neural_networks = None, mode="search"):
    game = Game()
    best_cell = Cell(np.float32(1000.0), rng.random((3,3)), np.array((300.0,300.0)), CELL_RADIUS, np.array([10, 50, 100, 500, 1000]), game)
    if mode == "test":
        best_cell.set_neural_network(*neural_networks)
    game.append_cells([best_cell])
    if mode != "test":
        if neural_networks is None:
            game.debug_randomize_cell_neural_network()
        else:
            best_cell.set_neural_network(*mutate_all_neural_network(neural_networks,rng.random() ** 2))


    game.set_energy_pool(FOOD_ENERGY * 200)

    cycle_count = 0
    max_cell_count = 0
    finish = False
    clock = pygame.time.Clock()
    running = True
    while running:
        game.fill_screen_white()
        game.cycle()
        cycle_count += 1
        max_cell_count = max(max_cell_count, game.get_cell_count())
        #if mode == "test":
        game.draw_cells()
        game.draw_foods()
        game.draw_rays()
        game.draw_APs()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finish = True
                running = False
                pygame.quit()
                return cycle_count, max_cell_count, best_cell.get_neural_network(), finish
        if game.get_cell_count() == 0:
            #print(f"cycle_count::{cycle_count}")
            #print(f"cell_count::{max_cell_count}")
            if mode != "test":
                best_neural_network = best_cell.get_neural_network()
                return cycle_count, max_cell_count, best_cell.get_neural_network(), finish
            running = False
            pygame.quit()

        if mode == "test":
            clock.tick(60)



if __name__ == "__main__":
    mode = "search"
    if mode == "search":
        bestNN = None
        best_cycleCount = 0
        best_cell_count = 0
        finish = False
        while finish is False:
            cycleCount, cell_count, NN, finish = main(bestNN, mode="search")
            if cycleCount*cell_count > best_cycleCount*best_cell_count:
                best_cycleCount = cycleCount
                best_cell_count = cell_count
                bestNN = NN
                print(f"now_best_cycleCount::{best_cycleCount}")
                print(f"now_best_cell_count::{best_cell_count}")
            else:
                print(f"cycleCount::{cycleCount}")
                print(f"cell_count::{cell_count}")

        print(f"best_cycleCount::{best_cycleCount}")
        print(f"best_cell_count::{best_cell_count}")
        with open("best_NN.pkl", "wb") as f:
            pickle.dump(bestNN, f)

    elif mode == "test":
        with open("best_NN.pkl", "rb") as f:
            bestNN = pickle.load(f)
        main(bestNN, mode="test")