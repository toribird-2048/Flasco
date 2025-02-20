import numpy as np
import pygame
from typing import Self


#誘因粒子：AttractantParticle:AP

rng = np.random.default_rng()


class Object:
    def __init__(self, energy:np.float32, colors:np.array(np.float32), position:np.array(np.float32), radius:np.int16):
        self.energy = np.float32(energy)
        self.colors = np.array(colors)
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
        velocity_vector = AP.get_info(2)
        self.move(velocity_vector)
        AP.true_remove_flag()


class Food(Object):
    def __init__(self, energy:np.float32, colors:np.array(np.float32), position:np.array(np.float32), radius:np.int16):
        super().__init__(energy, colors, position, radius)

    def __str__(self):
        return "Food"

class Cell(Object):
    def __init__(self, energy:np.float32, colors:np.array(np.float32), position:np.array(np.float32), radius:np.int16, neural_network:np.array(np.float32), valid_rays_count, timers_max_cycles:np.array(np.int16)):
        super().__init__(energy, colors, position, radius)
        self.neural_network = np.array(neural_network)
        self.rays = [None for _ in range(8)]
        self.valid_rays_count = valid_rays_count
        self.pre_position = np.array([0.0, 0.0])
        self.timers_max_cycles = np.array(timers_max_cycles)
        self.timers_cycle = [0, 0, 0, 0, 0]
        self.energy_absorption_rate = np.float32(0.01)
        self.tickly_energy_consumption = np.float32(0.05)

    def __str__(self):
        return "Cell"

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
        rays_info = [0 for _ in range(len(self.rays))]
        for k, ray in enumerate(self.rays[:self.valid_rays_count]):
            if ray is not None:
                rays_info[k] = ray.get_info()
        self_energy = self.energy
        self_delta_position = self.delta_position
        for timer_cycle in self.timers_cycle:
            pass

    def delta_position(self):
        """
        前フレームからの移動量
        :return:
        """
        return self.position - self.pre_position

    def AP_injector(self, theta:np.float32):
        """
        誘因粒子を角度{theta}で射出する
        :param theta:
        :return:
        """
        pass

    def ray_injector(self, theta:np.float32, injector_number:np.int8):
        """
        レイを角度{theta}で{injector_number}の射出機から発射
        :param theta:
        :param injector_number:
        :return:
        """
        pass

    def ray_info_receptor(self, ray:"Ray"):
        """
        レイが当たった際、情報を受け取る
        :param ray:
        :return:
        """
        pass


class Particle:
    def __init__(self, energy:np.float32, position:np.array(np.float32), velocity_vector:np.array(np.float32), lifetime:np.int16):
        self.energy = energy
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
        if info_type == "energy":
            return self.energy
        elif info_type == "position":
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
    def __init__(self, energy:np.float32, position:np.array(np.float32), velocity_vector:np.array(np.float32), lifetime:np.int16):
        super().__init__(energy, position, velocity_vector, lifetime)
        self.object_info = [0] * 5
        self.object_types = {"Cell":-1, "Food":0, "Unmovable":1}

    def get_type(self):
        #タイプ(識別子)は整数でなく、-1~1を分割したその点であらわす。1, 2, 3, 4, 5 -> -1, -0.5, 0, 0.5, 1
        #色も[-1,1]の範囲で渡す
        pass

    def set_theta(self, theta:np.float32):
        self.object_info[0] = theta / np.pi #正規化

    def set_object_info(self, objectA):
        object_colors = objectA.get_info("colors")
        object_type_str = str(objectA)
        object_type = self.object_types[object_type_str]
        self.object_info[1:] = *object_colors, object_type
        self.true_remove_flag()

class AttractantParticle(Particle):
    def __init__(self, energy:np.float32, position:np.array(np.float32), velocity_vector:np.array(np.float32), lifetime:np.int16):
        super().__init__(energy, position, velocity_vector, lifetime)


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
        self.food_set:list[Food] = []
        self.cell_set:list[Cell] = []
        self.AP_set:list[AttractantParticle] = []
        self.ray_set:list[Ray] = []
        self.energy_pool = np.float32(0.0)
        self.energy_per_food = np.float32(1.0)

    def append_cells(self,cells:list[Cell]):
        for cell in cells:
            self.cell_set.append(cell)

    def append_foods(self,foods:list[Food]):
        for food in foods:
            self.food_set.append(food)

    def set_energy_pool(self, energy):
        self.energy_pool = energy

    def generate_food(self): ###1
        food_count = int(np.floor(self.energy_pool / self.energy_per_food))
        for _ in range(food_count):
            food_position = rng.random(2) * self.world_size
            self.food_set.append(Food(self.energy_per_food, np.array([0.0, 1.0, 0.0]), food_position, np.int16(2)))
            self.energy_pool -= self.energy_per_food

    def rays_and_APs_touch_judgement(self): ###2
        for cell in self.cell_set:
            for ap in self.AP_set:
                if touch_judgement(ap, cell):
                    cell.AP_receptor(ap)
        for food in self.food_set:
            for ap in self.AP_set:
                if touch_judgement(ap, food):
                    food.AP_receptor(ap)
        for ray in self.ray_set:
            for cell in self.cell_set:
                if touch_judgement(ray, cell):
                    ray.set_object_info(cell)
            for food in self.food_set:
                if touch_judgement(ray, food):
                    ray.set_object_info(food)

    def calc_repulsion(self): ###3 repulsion:斥力
        for cell in self.cell_set:
            for sub_cell in self.cell_set:
                if sub_cell is not cell:
                    repulsion = calc_repulsion_between_cells(cell, sub_cell)
                    cell.premove(repulsion)

    def energy_absorbing(self): ###4
        for cell in self.cell_set:
            for sub_cell in self.cell_set:
                cell.absorb_energy(sub_cell)
            for food in self.food_set:
                cell.absorb_energy(food)

    def consume_cell_energy(self): ###5
        for cell in self.cell_set:
            cell.tickly_consume_energy()

    def update_cell_timers(self): ###6
        for cell in self.cell_set:
            cell.update_timers()

game = Game()
cell1 = Cell(np.float32(10.0), np.array([[1,1,1],[1,1,1],[1,1,1]]), np.array((0,0)), np.int16(5), [], 1, np.array([10, 2, 3, 4, 5]))
cell2 = Cell(np.float32(10.0), np.array([[1,1,1],[1,1,1],[1,1,1]]), np.array((1,0)), np.int16(5), [], 1, np.array([10, 2, 3, 4, 5]))
food1 = Food(np.float32(1.0), np.array([[1,1,1],[1,1,1],[1,1,1]]), np.array((1,1)), np.int16(3))

game.append_cells([cell1, cell2])
game.append_foods([food1])
game.set_energy_pool(10)

game.generate_food()
game.rays_and_APs_touch_judgement()  #これだけ大丈夫かわからん
game.calc_repulsion()
game.energy_absorbing()
game.consume_cell_energy()
game.update_cell_timers()












