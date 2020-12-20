import taichi as ti
import numpy as np
import random
import time

ti.init(arch=ti.gpu)

@ti.data_oriented
class sph_solver:
    def __init__(self, x_list, flag_list, gui, dim=2, **kwargs):
        # basic render settings
        self.dim = dim
        assert self.dim == 1 or self.dim == 2 or self.dim == 3
        self.dim_size = ti.Vector([1, 1.02])
        self.dt = 5e-4
        self.gui = gui

        # basic solver settings
        self.r = 5e-3  # particle spacing
        self.h = self.r * 2.0
        self.nbrs_num_max = 500
        self.grid_pnum_max = 500
        self.theta_c = 0.025
        self.theta_s = 0.0075
        self.density = 400
        self.snow_m = self.density * self.h ** 3
        self.mu_b = 1.0
        self.psi = 1.5
        self.E = 140
        self.mu = 0.2
        self.omega = 0.5
        self.epsilon = 10.0
        self.error_rate = 1e-3
        if 'theta_c' in kwargs:
            self.theta_c = kwargs['theta_c']
        if 'theta_s' in kwargs:
            self.theta_s = kwargs['theta_s']
        if 'density' in kwargs:
            self.density = kwargs['density']

        # inferenced settings
        self.p_num = len(x_list)
        self.grid_size = ti.ceil(self.dim_size / (2 * self.h)) + 1
        self.kernel_sig = 0.
        if self.dim == 1:
            self.kernel_sig = 2. / 3.
        elif self.dim == 2:
            self.kernel_sig = 10. / (7 * np.pi)
        elif self.dim == 3:
            self.kernel_sig = 1 / np.pi
        self.kernel_sig /= self.h ** self.dim

        # particle attributes
        self.x = ti.Vector(self.dim, ti.f32)  # positions
        self.v = ti.Vector(self.dim, ti.f32)  # velocity
        self.v_tmp = ti.Vector(self.dim, ti.f32)  # velocity_star
        self.rho = ti.var(ti.f32)  # density
        self.rho_0 = ti.var(ti.f32)  # rest density of the current time
        self.rho_tmp = ti.var(ti.f32)  # density_star
        self.a_other = ti.Vector(self.dim, ti.f32)  # acceleration_other
        self.a_friction = ti.Vector(self.dim, ti.f32)  # acceleration_friction
        self.a_lambda = ti.Vector(self.dim, ti.f32)  # acceleration_lambda
        self.a_G = ti.Vector(self.dim, ti.f32)  # acceleration_G
        self.pressure = ti.var(ti.f32)
        self.flag = ti.var(ti.f32)
        self.F_E = ti.Matrix(self.dim, self.dim, ti.f32)
        self.L = ti.Matrix(self.dim, self.dim, ti.f32)
        self.lbda = ti.var(ti.f32)
        self.G = ti.var(ti.f32)
        self.diag = ti.var(ti.f32)  # a_ii for jacobi solver
        self.vec_tmp = ti.Vector(self.dim, ti.f32)
        self.var_tmp = ti.var(ti.f32)
        self.lhs = ti.Vector(self.dim, ti.f32)
        self.rhs = ti.Vector(self.dim, ti.f32)
        self.mat_tmp = ti.Matrix(self.dim, self.dim, ti.f32)
        self.p_solve = ti.Vector(self.dim, ti.f32)
        self.v_solve = ti.Vector(self.dim, ti.f32)
        self.res_solve = ti.Vector(self.dim, ti.f32)
        self.s_solve = ti.Vector(self.dim, ti.f32)
        self.t_solve = ti.Vector(self.dim, ti.f32)
        ti.root.dense(ti.i, self.p_num).place(self.x, self.flag, self.v, self.v_tmp, self.rho, self.rho_0, self.rho_tmp,
                                              self.a_other, self.a_friction, self.a_lambda, self.a_G, self.pressure,
                                              self.F_E, self.L, self.lbda, self.G,
                                              self.diag, self.vec_tmp, self.var_tmp,
                                              self.lhs, self.rhs, self.mat_tmp, self.p_solve, self.v_solve,
                                              self.res_solve, self.s_solve, self.t_solve)
        self.nbrs_num = ti.var(ti.i32)
        self.nbrs_list = ti.var(ti.i32)
        nbrs_nodes = ti.root.dense(ti.i, self.p_num)
        nbrs_nodes.place(self.nbrs_num)
        nbrs_nodes.dense(ti.j, self.nbrs_num_max).place(self.nbrs_list)

        # grid attributes
        self.grid_p_num = ti.var(ti.i32)
        self.grids = ti.var(ti.i32)
        grid_nodes = ti.root.dense(ti.ij, (self.grid_size[0], self.grid_size[1]))
        grid_nodes.place(self.grid_p_num)
        grid_nodes.dense(ti.k, self.grid_pnum_max).place(self.grids)

        # data initialize
        self.rho.fill(self.density)
        self.parallel_init(np.array(x_list), np.array(flag_list))

    @ti.kernel
    def parallel_init(self, pos_list:ti.ext_arr(), flag_list:ti.ext_arr()):
        for i in range(self.p_num):
            for j in ti.static(range(self.dim)):
                self.x[i][j] = pos_list[i, j]
            self.flag[i] = flag_list[i]
            self.F_E[i] = ti.Matrix([[1, 0], [0, 1]])

    @ti.func
    def kernel(self, r):
        sig, h = ti.static(self.kernel_sig, self.h)
        q = r / h
        assert q >= 0.
        w = ti.cast(0.0, ti.f32)
        if q <= 1.:
            w = sig * (1. - 1.5 * q ** 2 + 0.75 * q ** 3)
        elif q <= 2.:
            w = sig * 0.25 * (2 - q) ** 3
        return w

    @ti.func
    def kernel_grad(self, r):
        sig, h = ti.static(self.kernel_sig, self.h)
        q = r / h
        assert q >= 0.0
        dw = ti.cast(0.0, ti.f32)
        if q <= 1.:
            dw = sig * (-3. * q + 2.25 * q ** 2)
        elif q <= 2.:
            dw = sig * -0.75 * (2 - q) ** 2
        return dw

    @ti.func
    def W(self, i, j):
        r = self.x[i] - self.x[j]
        w = self.kernel(r.norm())
        return w

    @ti.func
    def dW(self, i, j):
        r = self.x[i] - self.x[j]
        dw = self.kernel_grad(r.norm())
        return dw * r.normalized()

    @ti.func
    def bicgstab_vec_to_grad(self):
        # vec_tmp -> mat_tmp
        for i in self.x:
            if self.flag[i] == 0:
                grad_s = ti.Matrix([[0,0],[0,0]])
                grad_b = ti.Matrix([[0,0],[0,0]])
                for n in ti.static(range(self.nbrs_num[i])):
                    j = self.nbrs_list[i, n]
                    if self.flag[j] == 0:
                        grad_s += (self.v_tmp[j]-self.v_tmp[i]).outer_product(self.dW(i, j)) * self.snow_m / self.rho[j]
                    else:
                        grad_b += (self.v_tmp[j]-self.v_tmp[i]).outer_produce(self.dW(i, j)) / self.rho[j]
                grad_tilde = grad_s @ self.L[i].transpose() + (grad_b @ self.L[i].transpose()).trace() / 3.
                self.mat_tmp[i] = grad_tilde + (grad_b + grad_s).trace() / 3. - grad_tilde.trace() / 3.

    @ti.func
    def bicgstab_grad_to_vec(self):
        # nabla dot sig
        for i in self.x:
            if self.flag[i] == 0:
                self.vec_tmp[i].fill(0)
                for n in ti.static(range(self.nbrs_num[i])):
                    j = self.nbrs_list[i, n]
                    dW_ij = self.dW(i, j)
                    if self.flag[j] == 0:
                        self.vec_tmp[i] += self.mat_tmp[j] @ (self.L[j] @ dW_ij) * self.snow_m / self.rho[j]
                        self.vec_tmp[i] += self.mat_tmp[i] @ (self.L[i] @ dW_ij) * self.snow_m / self.rho[j]
                    else:
                        self.vec_tmp[i] += self.mat_tmp[i].trace() / 3. * (self.L[i] @ dW_ij) / self.rho[j]

    @ti.func
    def bicgstab_rmat_update(self):
        # mat_tmp -> vec_tmp
        for i in self.x:
            # update from grad to F_star
            self.mat_tmp[i] = self.F_E[i] + self.dt * self.mat_tmp[i] @ self.F_E[i]
            # F_star -> sig
            self.mat_tmp[i] = 2 * self.G[i] * ((self.mat_tmp[i] + self.mat_tmp[i].transpose()) / 2. - 1)

    @ti.func
    def bicgstab_lmat_update(self):
        for i in self.x:
            if self.flag[i] == 0:
                mat_prod = self.mat_tmp[i] @ self.F_E[i]
                self.mat_tmp[i] = self.G[i] * (mat_prod + mat_prod.transpose())

    @ti.kernel
    def update_neighbors(self):
        for p in self.x:
            nbrs_num = 0
            cell = (self.x[p] / (2. * self.h)).cast(int)
            for dI in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
                if all(0 <= cell + dI < self.grid_size):
                    for i in range(self.grid_p_num[cell + dI]):
                        nbr = self.grids[cell + dI, i]
                        if nbrs_num < self.nbrs_num_max and nbr != p and (self.x[p] - self.x[nbr]).norm() < 2 * self.h:
                            self.nbrs_list[p, nbrs_num] = nbr
                            nbrs_num += 1
            self.nbrs_num[p] = nbrs_num

    @ti.kernel
    def to_grid(self):
        for p in self.x:
            cell = (self.x[p] / (2. * self.h)).cast(int)
            cell_pnum = self.grid_p_num[cell].atomic_add(1)
            self.grids[cell, cell_pnum] = p

    @ti.kernel
    def print_grids(self):
        for i,j,k in self.grids:
            if self.grids[i, j, k] > 1500:
                print(i,j,k,self.grids[i,j,k])

    @ti.kernel
    def update_dt(self):
        # update velocity
        for p in self.x:
            if self.flag[p] == 0:
                self.v[p] += self.dt * (self.a_G[p] + self.a_other[p] + self.a_lambda[p] + self.a_friction[p])
        # update fe
        for p in self.x:
            if self.flag[p] == 0:
                grad_v = ti.Matrix([[0., 0.],[0., 0.]])
                for nbr in range(self.nbrs_num[p]):
                    i = self.nbrs_list[p, nbr]
                    if self.flag[i] == 0:
                        grad_v += self.v[i].outer_product(self.dW(i, p))
                self.F_E[p] += self.dt * grad_v @ self.F_E[p]
                U, Sigma, V = ti.svd(self.F_E[p], ti.f32)
                for n in ti.static(range(self.dim)):
                    Sigma[n, n] = min(1+self.theta_s, max(1-self.theta_c, Sigma[n, n]))
                self.F_E[p] = V @ Sigma @ V.transpose()
        # update position
        for p in self.x:
            if self.flag[p] == 0:
                self.x[p] += self.dt * self.v[p]

    @ti.kernel
    def add_graivity(self, g: ti.f32):
        for i in self.x:
            if self.flag[i] == 0:
                self.a_other[i][1] -= g

    @ti.kernel
    def check_dilation(self):
        dilation = 0
        for p in self.x:
            if self.x[p][1] < 0.:
                dilation+=1
                self.x[p][1] = 0.
                # self.v[p][1] = 0.
            # if any(self.x[p]>1) or any(self.x[p]<0):
            #     dilation += 1
            #     for n in ti.static(range(self.dim)):
            #         self.x[p][n] = min(max(self.x[p][n], 0.), 1.)
        # print("dilation = ", dilation)
        # print("snow density = ", self.rho[0])
        # print("boundary density = ", self.rho[self.p_num-1])
        # print("neighbors_num = ", self.nbrs_num[0])
        # print("neighbors = ", self.nbrs_list[0, max(self.nbrs_num[0]-1, 0)])
        # print(self.kernel(0.005))
        # print(self.kernel_sig)
        i = 0
        # print("acc_other:", self.a_other[i][0], self.a_other[i][1])
        # print("acc_friction:", self.a_friction[i][0], self.a_friction[i][1])
        # print("acc_lambda:", self.a_lambda[i][0], self.a_lambda[i][1])

    @ti.kernel
    def basic_solve(self):
        """friction & d_ii commented"""
        # update density & L for all particles
        # update rho for snow particles only
        for i in self.x:
            # prepare
            # compute a_other
            if self.flag[i] == 0 and self.x[i][1] <= 0.:
                self.a_other[i].fill(0)
            else:
                self.a_other[i][1] = -50
            m = 1 / (0.8 * self.h ** 2)  # for boundary particles, rho = 1 / V
            if self.flag[i] == 0:
                m = self.snow_m
                # self.a_friction[i] = self.v[i] + self.dt * self.a_other[i]
            self.rho[i] = 0.0
            self.L[i].fill(0)
            d_ii = 1.0
            # iteration
            for n in range(self.nbrs_num[i]):
                k = self.nbrs_list[i, n]
                # compute density
                if self.flag[k] == self.flag[i]:
                    self.rho[i] += self.W(i, k) * m
                mass_k = 1.0
                if self.flag[k] == 0:
                    mass_k = self.snow_m
                self.L[i] += mass_k / self.rho[k] * self.dW(i, k).outer_product(self.x[k] - self.x[i])
                if self.flag[i] == 0 and self.flag[k] != 0:
                    # compute friction only for boundary particles
                    # omit dynamic friction
                    x_ik = self.x[i] - self.x[k]
                    d_ii -= self.dt * self.mu_b * x_ik.dot(self.dW(i, k)) / self.rho[k] / (x_ik.norm() ** 2 + 0.01 * self.r ** 2)
            # post processing
            self.rho_0[i] = self.rho[i] * self.F_E[i].determinant()
            self.L[i] = self.L[i].inverse()
            # if self.flag[i] == 0:
                # self.a_friction[i] = (self.v[i] + self.dt * self.a_other[i]) / (self.dt * d_ii)
            # if i == 0:
            #     print(d_ii)
        # compute lame parameters
        for i in self.x:
            if self.flag[i] == 0:
                coef = ti.exp(self.epsilon * (self.rho_0[i]-self.density)/self.rho_0[i])
                self.G[i] = self.E / (2 * (1 + self.mu)) * coef
                self.lbda[i] = self.G[i] * self.mu * 2 / (1 - 2 * self.mu)

    @ti.kernel
    def implicit_state_solver(self):
        """
        v_star -> v_tmp
        rho_star -> rho_tmp
        a_ii -> diag
        grad_p -> vec_tmp
        Ap_i -> var_tmp
        """
        # PREPARE
        # compute v_star
        for i in self.x:
            if self.flag[i] == 0:
                self.v_tmp[i] = self.v[i] + self.dt * (self.a_other[i] + self.a_friction[i])
                # self.v_tmp[i] = self.v[i] + self.dt * self.a_other[i]
            else:
                self.v_tmp[i].fill(0)
        # compute rho_star and a_ii
        for i in self.x:
            if self.flag[i] == 0:
                self.rho_tmp[i] = self.rho[i]
                self.diag[i] = -self.rho_0[i]/self.lbda[i]
                for m in range(self.nbrs_num[i]):
                    k = self.nbrs_list[i, m]
                    p_mass = 1
                    if m == 0:
                        p_mass = self.snow_m
                    self.rho_tmp[i] -= self.dt * self.rho[i] * (self.v[k]-self.v[i]).dot(self.dW(i, k)) * p_mass / self.rho[k]
                    dw_ik = self.dW(i, k)
                    if self.flag[k] == 0:
                        self.diag[i] -= self.dt ** 2 * dw_ik.norm() ** 2 * self.snow_m ** 2 / self.rho[i] / self.rho[k]
                    for n in range(self.nbrs_num[i]):
                        b = self.nbrs_list[i, n]
                        if self.flag[b] == 0:
                            self.diag[i] -= self.dt ** 2 * self.dW(i, b).dot(dw_ik) * self.snow_m ** 2 / self.rho[b] / self.rho[k]
                        else:
                            self.diag[i] -= self.psi * self.dt ** 2 * self.dW(i, b).dot(dw_ik) * self.snow_m / self.rho[b] / self.rho[k]
        # SOLVE
        while True:
            error = 0
            # compute grad_p
            for i in range(self.p_num):
                if self.flag[i] == 0:
                    self.vec_tmp[i].fill(0)
                    for m in range(self.nbrs_num[i]):
                        j = self.nbrs_list[i, m]
                        if self.flag[j] == 0:
                            self.vec_tmp[i] += (self.pressure[i] + self.pressure[j]) * self.dW(i, j) * self.snow_m / self.rho[j]
                        else:
                            self.vec_tmp[i] += self.psi * self.pressure[i] * self.dW(i, j) / self.rho[j]
            # compute Ap and update
            for i in range(self.p_num):
                if self.flag[i] == 0:
                    self.var_tmp[i] = - self.rho_0[i] / self.lbda[i] * self.pressure[i] \
                                      + self.dt ** 2 * self.vec_tmp[i].dot(self.vec_tmp[i])
                    if abs((self.var_tmp[i] - self.rho_0[i] + self.rho_tmp[i])/(self.rho_0[i] - self.rho_tmp[i])) >= self.error_rate:
                        error += 1
            if error == 0:
                break
            for i in range(self.p_num):
                if self.flag[i] == 0:
                    self.pressure[i] += self.omega / self.diag[i] * (self.rho_0[i] - self.rho_tmp[i] - self.var_tmp[i])
        # compute a
        for i in self.x:
            if self.flag[i] == 0:
                self.a_lambda[i] = -self.vec_tmp[i]/self.rho[i]

    @ti.kernel
    def shear_deformation(self):
        # update v_star_star
        for i in self.x:
            if self.flag[i] == 0:
                self.v_tmp[i] = self.v[i] + self.dt * (self.a_other[i] + self.a_friction[i] + self.a_lambda[i])
        # compute rhs
        # -----------
        # copy into
        for i in self.x:
            if self.flag[i] == 0:
                self.vec_tmp[i] = self.v_tmp[i]
        # compute
        self.bicgstab_vec_to_grad()
        self.bicgstab_rmat_update()
        self.bicgstab_grad_to_vec()
        # copy out
        for i in self.x:
            if self.flag[i] == 0:
                self.rhs[i] = self.vec_tmp[i] / self.rho[i]
        # compute lhs
        # -----------
        # copy into
        for i in self.x:
            if self.flag[i] == 0:
                self.vec_tmp[i] = self.a_G[i]
        # compute
        self.bicgstab_vec_to_grad()
        self.bicgstab_lmat_update()
        self.bicgstab_grad_to_vec()
        for i in self.x:
            if self.flag[i] == 0:
                self.lhs[i] = self.a_G[i] - self.vec_tmp[i] * self.dt / self.rho[i]
        # initialize
        rho_prev = 1
        alpha = 1
        omega = 1
        for i in self.x:
            if self.flag[i] == 0:
                self.p_solve[i].fill(0)
                self.v_solve[i].fill(0)
                self.res_solve[i] = self.rhs[i] - self.lhs[i]
        # iteration
        while True:
            rho = 0.
            for i in self.x:
                if self.flag[i] == 0:
                    rho += self.res_solve[i].dot(self.rhs[i]-self.lhs[i])  # l1
            beta = (rho / rho_prev) * (alpha / omega)  # l2
            for i in self.x:
                if self.flag[i] == 0:
                    self.p_solve[i] = self.rhs[i] - self.lhs[i] + beta * (self.p_solve[i] - omega * self.v_solve[i])  # l3
                    self.vec_tmp[i] = self.p_solve[i]  # initialize for l4
            # l4
            self.bicgstab_vec_to_grad()
            self.bicgstab_lmat_update()
            self.bicgstab_grad_to_vec()
            for i in self.x:
                if self.flag[i] == 0:
                    self.v_solve[i] = self.p_solve[i] - self.dt ** 2 / self.rho[i] * self.vec_tmp[i]
            # l5
            alpha = 0
            for i in self.x:
                if self.flag[i] == 0:
                   alpha += self.res_solve[i].dot(self.v_solve[i])
            alpha = rho / alpha
            # l6
            for i in self.x:
                if self.flag[i] == 0:
                    self.a_G[i] += alpha * self.p_solve[i]
                    self.vec_tmp[i] = self.a_G[i]
            self.bicgstab_vec_to_grad()
            self.bicgstab_lmat_update()
            self.bicgstab_grad_to_vec()
            # l7
            error = 0
            for i in self.x:
                if self.flag[i] == 0:
                    Ax = self.a_G[i] - self.dt ** 2 / self.rho[i] * self.vec_tmp[i]
                    if (Ax - self.rhs[i]).norm >= 0.1:
                        error += 1
            if error == 0:
                break
            # l8
            for i in self.x:
                if self.flag[i] == 0:
                    self.s_solve[i] = self.rhs[i] - self.lhs[i] - alpha * self.v_solve[i]
                    self.vec_tmp[i] = self.s_solve[i]
            self.bicgstab_vec_to_grad()
            self.bicgstab_lmat_update()
            self.bicgstab_grad_to_vec()
            omega_up = 0
            omega_down = 0
            for i in self.x:
                if self.flag[i]:
                    self.t_solve[i] = self.s_solve[i] - self.vec_tmp[i] * self.dt ** 2 / self.rho[i]
                    omega_up += self.s_solve[i].dot(self.t_solve[i])
                    omega_down += self.t_solve[i].dot(self.t_solve[i])
            omega = omega_up / omega_down  # line 10
            for i in self.x:
                if self.flag[i]:
                    self.a_G[i] += omega * self.s_solve[i]  # line 11
                    self.vec_tmp[i] = self.a_G[i]
            self.bicgstab_vec_to_grad()
            self.bicgstab_lmat_update()
            self.bicgstab_grad_to_vec()
            error = 0
            for i in self.x:
                if self.flag[i]:
                    Ax = self.a_G[i] - self.dt ** 2 / self.rho[i] * self.vec_tmp[i]
                    self.lhs[i] = Ax
                    if (self.lhs[i]-self.rhs[i]).norm() >= 0.1:
                        error += 1
            if error == 0:
                break

    def solve(self):
        # refill neighbors
        self.grid_p_num.fill(0)
        self.nbrs_num.fill(-1)
        self.to_grid()
        self.update_neighbors()
        # solve
        self.basic_solve()
        # self.implicit_state_solver()
        self.update_dt()
        self.check_dilation()

    def render(self):
        snow_p = []
        pos_list = self.x.to_numpy()
        for i, pos in enumerate(pos_list):
            if self.flag[i] == 0:
                snow_p.append(pos)
        snow_p = np.array(snow_p)
        self.gui.circles(snow_p, radius=1.5, color=0xEEEEF0)
        self.gui.show()

    def save(self, i):
        snow_p = []
        pos_list = self.x.to_numpy()
        for j, pos in enumerate(pos_list):
            if self.flag[j] == 0:
                snow_p.append(pos)
        snow_p = np.array(snow_p)
        self.gui.circles(snow_p, radius=1.5, color=0xEEEEF0)
        filename = f'frame_{i:05d}.png'
        self.gui.show(filename)
        print(i)


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


def scene_init(snow_pnum, bound_dh):
    pos_list = []
    flag_list = []

    # snow particles: flag=0 (randomized initialization)
    add_particles_random([0.3, 0.05, 0.5, 0.25], snow_pnum // 3, pos_list, flag_list, flag=0)
    add_particles_random([0.4, 0.37, 0.6, 0.57], snow_pnum // 3, pos_list, flag_list, flag=0)
    add_particles_random([0.5, 0.69, 0.7, 0.89], snow_pnum // 3, pos_list, flag_list, flag=0)

    # bound particles: flag=1 (regular initialization)
    add_particles([0., 0.0, 1., bound_dh], bound_dh, bound_dh, pos_list, flag_list, flag=1)
    # add_particles([0., 0., 0., 1.], bound_dh, bound_dh, pos_list, flag_list, flag=1)
    # add_particles([1., 0., 1., 1.], bound_dh, bound_dh, pos_list, flag_list, flag=1)

    return pos_list, flag_list


def mpm_main():

    gui = ti.GUI('sph2d', res=512, background_color=0x112F41)
    pos_list, flag_list = scene_init(9000, 5e-3)
    test_solver = sph_solver(pos_list, flag_list, gui)

    # while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    start = time.time()
    for i in range(600):
        test_solver.solve()
    # test_solver.print_grids()
    # print(test_solver.grid_size)
        test_solver.save(i)
    print(600/(time.time()-start))

if __name__ == '__main__':
    mpm_main()