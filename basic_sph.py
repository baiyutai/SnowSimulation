import taichi as ti
import numpy as np
import random
import time

ti.init(arch=ti.gpu)

@ti.data_oriented
class sph_solver:
    def __init__(self, x_list, gui, dim=2, **kwargs):
        # basic render settings
        self.dim = dim
        self.dim_size = ti.Vector([1., 1.])
        self.minx = ti.Vector([-1., -1.])
        assert self.dim == 1 or self.dim == 2 or self.dim == 3
        self.dt = 2e-3  # time unit is 1/30 second
        # self.dt = 0.022
        self.gui = gui
        self.vel_max = 50

        # basic solver settings
        self.r = 0.1  # particle spacing
        self.h = self.r
        self.nbrs_num_max = 3000
        self.grid_pnum_max = 3000
        self.g = ti.Vector([0, -9.8])
        # check section 5.3 for setting
        self.sigma = 0.3
        self.beta = 0.3  # a non-zero value
        self.gamma = 0.1  # typically 0 ~ 0.2
        self.alpha = 0.3
        # check section 7 at the last for setting
        self.k = 0.504
        self.k_near = 5.04
        self.k_spring = 0.3
        self.rho_0 = 100.0
        # see section 6.1
        self.mu = 0
        # see https://github.com/omgware/fluid-simulator-v2/blob/master/fluid-simulator/src/com/fluidsimulator/FluidSimulatorSPH.java
        self.collisionForce = 100.0

        # inferenced settings
        self.p_num = len(x_list)
        self.grid_size = ti.ceil((self.dim_size - self.minx) / (2 * self.h)) + 10

        # particle attributes
        self.x = ti.Vector(self.dim, ti.f32)  # positions
        self.x_old = ti.Vector(self.dim, ti.f32)  # old positions
        self.v = ti.Vector(self.dim, ti.f32)  # velocity
        ti.root.dense(ti.i, self.p_num).place(self.x, self.x_old, self.v)

        self.nbrs_num = ti.var(ti.i32)
        self.nbrs_list = ti.var(ti.i32)
        self.strs_list = ti.var(ti.f32)
        self.strs_flag = ti.var(ti.i32)
        # self.L_list = ti.var(ti.f32)
        nbrs_nodes = ti.root.dense(ti.i, self.p_num)
        nbrs_nodes.place(self.nbrs_num)
        nbrs_nodes.dense(ti.j, self.nbrs_num_max).place(self.nbrs_list, self.strs_list, self.strs_flag)

        # grid attributes
        self.grid_p_num = ti.var(ti.i32)
        self.grids = ti.var(ti.i32)
        grid_nodes = ti.root.dense(ti.ij, (self.grid_size[0], self.grid_size[1]))
        grid_nodes.place(self.grid_p_num)
        grid_nodes.dense(ti.k, self.grid_pnum_max).place(self.grids)

        self.particle_list = np.array(x_list)

    @ti.kernel
    def init(self, p_list:ti.ext_arr()):
        for i in range(self.p_num):
            for j in ti.static(range(self.dim)):
                self.x_old[i][j] = p_list[i, j]
                self.x[i][j] = p_list[i, j]
            # self.v[i][0] = ti.random() - 0.5

    @ti.kernel
    def update_neighbors(self):
        for p in self.x:
            nbrs_num = 0
            cell = ((self.x[p] - self.minx) / (2. * self.h)).cast(int)
            for dI in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
                if all(0 <= cell + dI < self.grid_size):
                    for i in range(self.grid_p_num[cell + dI]):
                        nbr = self.grids[cell + dI, i]
                        if nbrs_num < self.nbrs_num_max and nbr != p and (self.x[p] - self.x[nbr]).norm() < 2 * self.h:
                            self.nbrs_list[p, nbrs_num] = nbr
                            self.strs_flag[p, nbrs_num] = 0
                            nbrs_num += 1
            self.nbrs_num[p] = nbrs_num

    @ti.kernel
    def to_grid(self):
        for p in self.x:
            cell = ((self.x[p]-self.minx) / (2. * self.h)).cast(int)
            cell_pnum = self.grid_p_num[cell].atomic_add(1)
            self.grids[cell, cell_pnum] = p

    @ti.kernel
    def apply_viscosity(self):
        for i in self.x:
            for nbr in range(self.nbrs_num[i]):
                j = self.nbrs_list[i, nbr]
                if i < j:
                    r = self.x[i] - self.x[j]
                    q = r.norm() / self.h
                    if q < 1:
                        u = (self.v[i]-self.v[j]).dot(r)
                        if u > 0:
                            I = self.dt * (1-q) * (self.sigma * u + self.beta * u ** 2) * r
                            self.v[i] -= I / 2
                            self.v[j] += I / 2

    @ti.kernel
    def pos2old(self):
        for i in self.x:
            self.x_old[i] = self.x[i]
            self.x[i] += self.dt * self.v[i]

    @ti.kernel
    def double_density_relaxation(self):
        for i in self.x:
            rho = 0
            rho_near = 0
            for nbr in range(self.nbrs_num[i]):
                j = self.nbrs_list[i, nbr]
                q = (self.x[i] - self.x[j]).norm() / self.h
                if q < 1:
                    rho += (1-q) ** 2
                    rho_near += (1-q) ** 3
            P = self.k * (rho - self.rho_0)
            P_near = self.k_near * rho_near
            dx = ti.Vector([0., 0.])
            for nbr in range(self.nbrs_num[i]):
                j = self.nbrs_list[i, nbr]
                q = (self.x[i] - self.x[j]).norm() / self.h
                if q < 1:
                    D = self.dt ** 2 * (P * (1-q) + P_near * (1-q)**2) * (self.x[i]-self.x[j])
                    self.x[j] += D / 2
                    dx -= D / 2
            self.x[i] += dx

    @ti.kernel
    def set_strings(self):
        for i in self.x:
            for nbr in range(self.nbrs_num[i]):
                j = self.nbrs_list[i, nbr]
                if i < j:
                    if self.strs_flag[i, nbr] == 0:
                        self.strs_flag[i, nbr] = 1
                        self.strs_list[i, nbr] = self.h
                    r = (self.x[i] - self.x[j]).norm()
                    d = self.gamma * self.strs_list[i, nbr]
                    if r > self.strs_list[i, nbr] + d:
                        self.strs_list[i, nbr] += self.dt * self.alpha * (r - self.strs_list[i, nbr] - d)
                    elif r < self.strs_list[i, nbr] - d:
                        self.strs_list[i, nbr] -= self.dt * self.alpha * (self.strs_list[i, nbr] - d - r)
                    # if self.strs_list[i, nbr] < self.h:
                        # self.strs_flag[i, nbr] = 0

    @ti.kernel
    def displace_strings(self):
        for i in self.x:
            for nbr in range(self.nbrs_num[i]):
                j = self.nbrs_list[i, nbr]
                if i < j and self.strs_flag[i, nbr] != 0:
                    r = self.x[i] - self.x[j]
                    L = self.strs_list[i, nbr]
                    D = self.dt ** 2 * self.k_spring * (1 - L/self.h) * (L - r.norm()) * r
                    self.x[i] -= D / 2
                    self.x[j] += D / 2

    @ti.kernel
    def basic_solve(self):
        for i in self.x:
            self.v[i] = (self.x[i] - self.x_old[i])/self.dt
            if 0 < self.x[i][0]:
                self.v[i] += self.g * self.dt  # apply gravity
            # wall collision
            tempVect = ti.Vector([0., 0.])
            if self.x[i][0] > 1:
                tempVect[0] += 1 - self.x[i][0]
            if self.x[i][0] < 0:
                # self.x[i][0] = 0
                tempVect[0] += 0 - self.x[i][0]
            if self.x[i][1] > 1:
                tempVect[1] += 1 - self.x[i][1]
            if self.x[i][1] < 0:
                # self.x[i][1] = 0
                tempVect[1] += 0 - self.x[i][1]
            self.v[i] += tempVect * self.collisionForce
            # cap velocity
            for j in ti.static(range(self.dim)):
                self.v[i][j] = min(max(self.v[i][j], -self.vel_max), self.vel_max)

    def rehash(self):
        self.grid_p_num.fill(0)
        self.nbrs_num.fill(-1)
        self.to_grid()
        self.update_neighbors()

    def solve_html(self):
        self.apply_viscosity()
        self.pos2old()
        self.rehash()
        self.set_strings()
        self.displace_strings()
        self.double_density_relaxation()
        self.basic_solve()


    def render(self):
        # snow_p = []
        pos_list = self.x.to_numpy()
        # for i, pos in enumerate(pos_list):
        #     if self.flag[i] == 0:
        #         snow_p.append(pos)
        # snow_p = np.array(snow_p)
        self.gui.circles(pos_list, radius=2.0, color=0xEEEEF0)
        self.gui.show()

    def save(self, i):
        pos_list = self.x.to_numpy()
        self.gui.circles(pos_list, radius=2.0, color=0xEEEEF0)
        filename = f'frame_{i:05d}.png'
        self.gui.show(filename)


def add_particles(pos_bound, dx, dy, pos_list, flag_list, flag=0):
    e = 1e-5
    xl, yl, xr, yr = pos_bound
    for x in np.arange(xl, xr, dx):
        for y in np.arange(yl, yr, dy):
            pos_list.append([x, y])
            flag_list.append(flag)


def add_particles_random(pos_bound, p_num, pos_list, flag_list, flag=0):
    xl, yl, xr, yr = pos_bound
    for i in range(p_num):
        pos_list.append([xl+random.random()*(xr-xl), yl+random.random()*(yr-yl)])
        flag_list.append(flag)


def scene_init(snow_pnum):
    pos_list = []
    flag_list = []

    # snow particles: flag=0 (randomized initialization)
    add_particles_random([0.3, 0.05, 0.5, 0.25], snow_pnum // 3, pos_list, flag_list, flag=0)
    add_particles_random([0.4, 0.25, 0.6, 0.45], snow_pnum // 3, pos_list, flag_list, flag=0)
    add_particles_random([0.5, 0.45, 0.7, 0.65], snow_pnum // 3, pos_list, flag_list, flag=0)

    # bound particles: flag=1 (regular initialization)
    # add_particles([0., 0.0, 1., bound_dh], bound_dh, bound_dh, pos_list, flag_list, flag=1)
    # add_particles([0., 0., 0., 1.], bound_dh, bound_dh, pos_list, flag_list, flag=1)
    # add_particles([1., 0., 1., 1.], bound_dh, bound_dh, pos_list, flag_list, flag=1)

    return pos_list, flag_list


def mpm_main():

    gui = ti.GUI('sph2d', res=512, background_color=0x112F41)
    pos_list, _ = scene_init(9000)
    test_solver = sph_solver(pos_list, gui)
    test_solver.init(test_solver.particle_list)

    # while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    start = time.time()
    for i in range(600):
        # print("solving...")
        for j in range(1):
            test_solver.solve_html()
        # test_solver.render()
        # break
        test_solver.save(i)
        print("{} images saved".format(i))
    print(1200/(time.time()-start))


if __name__ == '__main__':
    mpm_main()