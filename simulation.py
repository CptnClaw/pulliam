import random
from math import floor
from numpy.random import default_rng

P_max = 0.2
B_max = 20
OPT_E1 = 7
OPT_E2 = 3
A1 = 0.2
A2 = 0.5
k = 100
GRID_WIDTH = 20
GRID_HEIGHT = 20
E1_VALUES = list(range(1, 9+1))  # 1 ... 9
E2_VALUES = list(range(1, 10+1))  # 1 ... 10
SIM_LENGTH = 100
REPETITIONS = 50

RED = '\033[0;41m'
YELLOW = '\033[0;43m'
CRYSTAL ='\033[0;46m'
BLUE = '\033[0;44m'
BLACK = '\033[0;37m'

def p(e1):
    deviation = (e1 - OPT_E1)**2
    res = P_max * (1 - (A1 * deviation))
    return max(res, 0)

def b(e2):
    deviation = (e2 - OPT_E2)**2
    res = B_max * (1 - (A2 * deviation))
    return max(res, 0)

rng = default_rng()

class Cell:

    def __init__(self, e1=None, e2=None):
        if e1 == None:
            e1 = rng.choice(E1_VALUES)
        if e2 == None:
            e2 = rng.choice(E2_VALUES)
        self.e1 = e1
        self.e2 = e2
        self.population = 0
        self.next_gen = 0
        self._calculate()

    def _calculate(self):
        self.p = p(self.e1)
        self.b = b(self.e2)
        self.lam = self.p * self.b
        
    def effective_p(self):
        carrying_capacity = max(0, 1 - (self.population / float(k)))
        return carrying_capacity * self.p

    def color(self):
        if self.lam > 8:
            return RED
        if self.lam > 2:
            return YELLOW
        if self.lam > 1:
            return BLUE
        if self.lam > 0:
            return CRYSTAL
        return BLACK

class Grid:
    def __init__(self, is_ecological):
        if not is_ecological:
            self.array = [[Cell() for col in range(GRID_WIDTH)]\
                            for row in range(GRID_HEIGHT)]
        else:
            self.array = [[Cell(e1, e2) for e1 in E1_VALUES]\
                            for e2 in E2_VALUES]
    
    def next_gen_size(self):
        return sum([sum([cell.next_gen for cell in row]) for row in self.array])

    def pop_size(self):
        return sum([sum([cell.population for cell in row]) for row in self.array])

    def stats(self):
        num_sources = 0
        num_occupied_sources = 0
        pop_in_sources = 0
        pop_size = self.pop_size()
        for row in self.array:
            for cell in row:
                if cell.lam > 1:
                    num_sources += 1
                    if cell.population > 0:
                        num_occupied_sources += 1
                        pop_in_sources += cell.population
        if pop_size == 0:
            return [num_sources, pop_size]
        occupied_sources_freq = int(100*float(num_occupied_sources) / num_sources)
        pop_in_sources_freq = int(100*float(pop_in_sources) / pop_size)
        return [num_sources, pop_size, occupied_sources_freq, pop_in_sources_freq]

    def repr_lam(self):
        res = '\nLambda\n'
        for row in self.array:
            res += '|'.join(['%.1f' % cell.lam for cell in row])
            res += '\n'
        return res

    def repr_sourcesink(self):
        def sourcesink(lam):
            if lam > 1:
                return '+'
            if lam < 1:
                return '-'
            return '0'

        res = '\nSource/Sink\n'
        for row in self.array:
            res += ','.join([sourcesink(cell.lam) for cell in row])
            res += '\n'

        res += str(sum([sum([1 for cell in row if cell.lam > 1]) for row in self.array]))
        return res

    def repr_population(self):
        res = '\nPopulation\n' 
        for row in self.array:
            res += '|'.join(['%s%2d%s' % \
                    (cell.color(), cell.population, BLACK)\
                    for cell in row])
            res += '\n'
        res += "Size: %d" % self.pop_size()
        return res

    def repr_next_gen(self):
        res = '\nNext Gen\n'
        for row in self.array:
            res += '|'.join(['%s%2d%s' % \
                    (cell.color(), cell.next_gen, BLACK)\
                    for cell in row])
            res += '\n'
        res += "Next size: %d" % self.next_gen_size()
        return res

    def __repr__(self):
        res = ''
        res += self.repr_population()
        return res

class Simulation:
    def __init__(self, sigma):
        #self.ecology = Grid(is_ecological=True)
        self.world = Grid(is_ecological=False)
        self.sigma = sigma

        # Initialize population in source tiles
        for row in self.world.array:
            for cell in row:
                if cell.lam > 1:
                    cell.population = 5

    def birth_and_migrate(self):
        for i, row in enumerate(self.world.array):
            for j, cell in enumerate(row):
                for adult in range(cell.population):
                    offsprings = rng.poisson(cell.b)
                    for seed in range(offsprings):
                        spread = rng.exponential(1 / self.sigma)
                        spread_dir = rng.choice([[0,1],[0,-1],[1,0],[-1,0]])
                        a = (0.5 + spread) * spread_dir[0]
                        b = (0.5 + spread) * spread_dir[1]
                        spread_dir[0] = int((0.5 + spread) * spread_dir[0])
                        spread_dir[1] = int((0.5 + spread) * spread_dir[1])
                        new_i = i + spread_dir[0]
                        new_j = j + spread_dir[1]
                        if new_i > -1 and new_i < GRID_HEIGHT and \
                            new_j > -1 and new_j < GRID_WIDTH:
                                self.world.array[new_i][new_j].next_gen += 1

    def grow(self):
        #print(self.world.repr_next_gen())
        for row in self.world.array:
            for cell in row:
                survived = rng.binomial(cell.next_gen, cell.effective_p())
                cell.population = survived
                cell.next_gen = 0

    def step(self):
        self.birth_and_migrate()
        self.grow()

    def simulate(self):
        for i in range(SIM_LENGTH):
            self.step()
            #print(self.world)

class Simulator:
    def run(sigmas):
        print('sigma,num_sources,pop_size,occupied_sources_freq,pop_in_sources_freq')
        for sigma in sigmas:
            for i in range(REPETITIONS):
                sim = Simulation(sigma)
                sim.simulate()
                stats = sim.world.stats()
                print(','.join(str(val) for val in [sigma]+stats))

if __name__ == '__main__':
    Simulator.run([1, 2, 4, 8, 16, 32])
