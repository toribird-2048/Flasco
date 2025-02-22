import numpy as np
import pygame
from typing import Self
import copy


#誘因粒子：AttractantParticle:AP

rng = np.random.default_rng()


class Object:
    def __init__(self, energy:np.float32, colors:np.array(np.float32), position:np.array(np.float32), radius:np.int16):
        self.energy = np.float32(energy)
        self.colors = np.array(colors) #(R, G, B)それぞれ0~1
        self.position = np.array(position)
        self.radius = np.int16(radius)
        self.pre_movement = np.array([0.0, 0.0])
        self.remove_flag = False

    def __str__(self):
        return "Object"

    def get_info(self, info_type:str):
        """
        :return: energy, colors, position, radius, pre_movement, remove_flag
        """
        if info_type == "energy":
            return self.energy
        elif info_type == "colors":
            return self.colors
        elif info_type == "position":
            return self.position
        elif info_type == "radius":
            return self.radius
        elif info_type == "pre_movement":
            return self.pre_movement
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
        self.position += vector
    
    def no_energy_check(self):
        """
        エネルギーが0になったらremove_flagをTrueにする
        :return:
        """
        if self.energy <= 0:
            self.true_remove_flag()

    def edit_energy(self, value:np.float32):
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
        transferring_energy = self.edit_energy(-value)
        target.edit_energy(-transferring_energy)

    def AP_receptor(self, AP):
        """
        誘因粒子が当たった際に関数が呼ばれ、オブジェクトを動かす。
        :param AP:
        :return:
        """
        velocity_vector = AP.get_info("velocity_vector")
        self.pre_movement += velocity_vector
        AP.true_remove_flag()


class Food(Object):
    def __init__(self, energy:np.float32, colors:np.array(np.float32), position:np.array(np.float32), radius:np.int16):
        super().__init__(energy, colors, position, radius)

    def __str__(self):
        return "Food"

class Cell(Object):
    def __init__(self, energy:np.float32, colors:np.array(np.float32), position:np.array(np.float32), radius:np.int16, valid_rays_count, timers_max_cycles:np.array(np.int16), game:"Game"):
        super().__init__(energy, colors, position, radius)
        self.rays = [None for _ in range(8)]
        self.valid_rays_count = valid_rays_count
        self.pre_position = np.array([0.0, 0.0])
        self.timers_max_cycles = np.array(timers_max_cycles)
        self.timers_cycle = [0, 0, 0, 0, 0]
        self.energy_absorption_rate = np.float32(0.01)
        self.tickly_energy_consumption = np.float32(0.05)
        self.weight_input_to_hidden = np.zeros((30,48))
        self.bias_input_to_hidden = np.zeros((30,1))
        self.weight_hidden_to_output = np.zeros((21,30))
        self.bias_hidden_to_output = np.zeros((21,1))
        self.neural_network_outputs = np.zeros((21, 1))
        self.game = game

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

    def premove(self, vector:np.array(np.float32)):
        self.pre_movement = vector

    def absorb_energy(self, objectA):
        self.transfer_energy(objectA, -self.energy_absorption_rate)

    def tickly_consume_energy(self):
        self.energy -= self.tickly_energy_consumption

    def update_timers(self):
        for k, timer in enumerate(self.timers_cycle):
            self.timers_cycle[k] = np.mod(self.timers_cycle[k]+1, self.timers_max_cycles[k])

    def info_collector_for_NN(self):
        """
        ニューラルネットワークへ渡す情報を集める
        :return:
        """
        rays_info = np.zeros((8,5), dtype=np.float32)
        for k, ray in enumerate(self.rays[:self.valid_rays_count]):
            if ray is not None:
                rays_info[k] = ray.get_ray_info()
        rays_info = np.ravel(rays_info)
        self_energy = self.energy
        self_delta_position = self.delta_position()
        timers_info = self.timers_cycle
        info_for_NN = np.concatenate((rays_info, np.array([self_energy, *self_delta_position]), timers_info), axis=0)
        info_for_NN = info_for_NN.reshape((info_for_NN.size, 1))
        return info_for_NN[:]

    def calc_neural_network(self, NN_input):
        input_layer:np.array(np.float32) = np.array(NN_input).reshape((48,1))
        hidden_layer:np.array(np.float32) = np.tanh(self.weight_input_to_hidden @ input_layer + self.bias_input_to_hidden)
        output_layer:np.array(np.float32) = np.tanh(self.weight_hidden_to_output @ hidden_layer + self.bias_hidden_to_output)
        self.neural_network_outputs = output_layer.reshape(output_layer.size)

    def delta_position(self):
        """
        前フレームからの移動量
        :return:
        """
        return self.position - self.pre_position

    def shoot_rays(self):
        neural_network_outputs_ray = self.neural_network_outputs[:16]
        ray_shoot_flags = [neural_network_outputs_ray[k*2] for k in range(8)]
        ray_shoot_theta = [neural_network_outputs_ray[k*2+1] * np.pi for k in range(8)]
        for k, ray in enumerate(self.rays):
            if ray is None:
                if ray_shoot_flags[k] > 0:
                    self.rays[k] = Ray(self.position, np.array([np.cos(ray_shoot_theta[k]), np.sin(ray_shoot_theta[k])]), np.int16(20))
        return self.rays

    def shoot_AP(self):
        neural_network_outputs_AP = self.neural_network_outputs[16:18]
        AP_shoot_flag = neural_network_outputs_AP[0]
        AP_shoot_theta = neural_network_outputs_AP[1] * np.pi
        return AttractantParticle(self.position, np.array((np.cos(AP_shoot_theta), np.sin(AP_shoot_theta))), np.int16(20))

    def duplicate(self):
        neural_network_outputs_duplicate = self.neural_network_outputs[18:21]
        duplicate_flag = neural_network_outputs_duplicate[0]
        duplicate_energy_ratio = neural_network_outputs_duplicate[1:3]
        print(duplicate_energy_ratio)
        duplicate_energy_ratio = duplicate_energy_ratio / np.sum(duplicate_energy_ratio + 1 + 1e-10)
        if duplicate_flag > 0:
            cell1_energy = self.energy * duplicate_energy_ratio[0]
            cell2_energy = self.energy * duplicate_energy_ratio[1]
            random_position = self.position + 0.5 * rng.random((2,1))
            random_position = np.mod(random_position, self.game.world_size)
            cell2 = Cell(cell2_energy, self.colors, self.position, self.radius, self.valid_rays_count, self.timers_max_cycles,self.game)
            cell2.set_neural_network(self.weight_input_to_hidden, self.bias_input_to_hidden, self.weight_hidden_to_output, self.bias_hidden_to_output)
            return cell2
        else:
            return None



class Particle:
    def __init__(self, position:np.array(np.float32), velocity_vector:np.array(np.float32), lifetime:np.int16):
        self.position = position
        self.velocity_vector = velocity_vector
        self.lifetime = lifetime
        self.age = 0
        self.remove_flag = False

    energy_consumption = np.float32(1.0)
    particle_speed = np.float32(10.0)

    def get_info(self, info_type:str):
        """
        :return: energy, position, velocity_vector, lifetime, age, self.remove_flag
        """
        if info_type == "position":
            return self.position
        elif info_type == "velocity_vector":
            return self.velocity_vector
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

    def move(self):
        """
        決まった方向(velocity_vector)に動く
        :return:
        """
        self.position += self.velocity_vector

    def lifetime_check(self):
        """
        粒子の寿命チェック。寿命を過ぎたらremove_flagをTrueにする
        :return:
        """
        if self.lifetime <= self.age:
            self.remove_flag = True

class Ray(Particle):
    def __init__(self, position:np.array(np.float32), velocity_vector:np.array(np.float32), lifetime:np.int16):
        super().__init__(position, velocity_vector, lifetime)
        self.object_info_and_theta = np.zeros(5)
        self.object_types = {"Cell":np.float32(-1.0), "Food":np.float32(0.0), "Unmovable":np.float32(1.0)}

    def get_ray_info(self):
        return self.object_info_and_theta

    def set_theta(self, theta:np.float32):
        self.object_info_and_theta[0] = theta / np.pi #正規化

    def set_object_info(self, objectA):
        object_colors = objectA.get_info("colors")
        object_type_str = str(objectA)
        object_type = self.object_types[object_type_str]
        self.object_info_and_theta[1:] = *object_colors, object_type
        self.true_remove_flag()

class AttractantParticle(Particle):
    def __init__(self, position:np.array(np.float32), velocity_vector:np.array(np.float32), lifetime:np.int16):
        super().__init__(position, velocity_vector, lifetime)


def touch_judgement(particle:Particle, objectA:Object):
    particle_position = particle.get_info("position")
    objectA_position = objectA.get_info("position")
    objectA_radius = objectA.get_info("radius")
    if np.linalg.norm(particle_position - objectA_position) <= objectA_radius:
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



class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode(np.array((600, 600)))
        self.world_size = np.array((600, 600))#原点は左下
        self.clock = pygame.time.Clock()
        self.running = True
        self.food_list:list[Food] = []
        self.cell_list:list[Cell] = []
        self.AP_list:list[AttractantParticle] = []
        self.ray_list:list[Ray] = []
        self.energy_pool = np.float32(0.0)
        self.energy_per_food = np.float32(1.0)

    def append_cells(self,cells:list[Cell]):
        for cell in cells:
            self.cell_list.append(cell)

    def append_foods(self,foods:list[Food]):
        for food in foods:
            self.food_list.append(food)

    def debug_append_APs(self,APs:list[AttractantParticle]):
        for ap in APs:
            self.AP_list.append(ap)

    def debug_append_rays(self,rays:list[Ray]):
        for ray in rays:
            self.ray_list.append(ray)

    def set_energy_pool(self, energy):
        self.energy_pool = energy

    def debug_randomize_cell_neural_network(self):
        for cell in self.cell_list:
            cell.set_neural_network_random()

    def generate_food(self): ###1
        food_count = int(np.floor(self.energy_pool / self.energy_per_food))
        for _ in range(food_count):
            food_position = rng.random(2) * self.world_size
            self.food_list.append(Food(self.energy_per_food, np.array([0.0, 1.0, 0.0]), food_position, np.int16(2)))
            self.energy_pool -= self.energy_per_food

    def rays_and_APs_touch_judgement(self): ###2
        for ap in self.AP_list:
            for cell in self.cell_list:
                    if touch_judgement(ap, cell):
                        cell.AP_receptor(ap)
            for food in self.food_list:
                    if touch_judgement(ap, food):
                        food.AP_receptor(ap)
        for ray in self.ray_list:
            for cell in self.cell_list:
                if touch_judgement(ray, cell):
                    ray.set_object_info(cell)
            for food in self.food_list:
                if touch_judgement(ray, food):
                    ray.set_object_info(food)

    def calc_repulsion(self): ###3 repulsion:斥力
        for cell in self.cell_list:
            for sub_cell in self.cell_list:
                if sub_cell is not cell:
                    repulsion = calc_repulsion_between_cells(cell, sub_cell)
                    cell.premove(repulsion)

    def energy_absorbing(self): ###4
        for cell in self.cell_list:
            for sub_cell in self.cell_list:
                cell.absorb_energy(sub_cell)
            for food in self.food_list:
                cell.absorb_energy(food)

    def consume_cell_energy(self): ###5
        for cell in self.cell_list:
            cell.tickly_consume_energy()

    def update_cell_timers(self): ###6
        for cell in self.cell_list:
            cell.update_timers()

    def execute_neural_network(self): ###7
        for cell in self.cell_list:
            collected_info = cell.info_collector_for_NN()
            cell.calc_neural_network(collected_info)

    def shoot_rays(self): ###8
        for cell in self.cell_list:
            rays = cell.shoot_rays()
            self.ray_list += [ray for ray in rays if ray is not None]

    def shoot_APs(self): ###9
        for cell in self.cell_list:
            AP = cell.shoot_AP()
            self.AP_list.append(AP)

    def cells_duplicate(self): ###10
        cell_list = self.cell_list[:]
        for cell in self.cell_list:
            cell2 = cell.duplicate()
            if cell2 is not None:
                cell_list.append(cell2)
        self.cell_list = cell_list




game = Game()
cell1 = Cell(np.float32(10.0), np.array([[1,1,1],[1,1,1],[1,1,1]]), np.array((0.0,0.0)), np.int16(5), 1, np.array([10, 2, 3, 4, 5]), game)
#cell2 = Cell(np.float32(10.0), np.array([[1,1,1],[1,1,1],[1,1,1]]), np.array((1,0)), np.int16(5), 1, np.array([10, 2, 3, 4, 5]), game)
food1 = Food(np.float32(1.0), np.array([[1,1,1],[1,1,1],[1,1,1]]), np.array((1.0,1.0)), np.int16(3))

ap1 = AttractantParticle(np.array((0.5,0.5)), np.array((0.1,0.1)), np.int16(10))

game.append_cells([cell1])
game.append_foods([food1])
game.debug_append_APs([ap1])
game.set_energy_pool(10)

game.generate_food()
game.rays_and_APs_touch_judgement()
game.calc_repulsion()
game.energy_absorbing()
game.consume_cell_energy()
game.update_cell_timers()
game.debug_randomize_cell_neural_network()
game.execute_neural_network()
game.shoot_rays()
game.shoot_APs()
game.cells_duplicate()

print(cell1.neural_network_outputs)
#print(game.ray_list)