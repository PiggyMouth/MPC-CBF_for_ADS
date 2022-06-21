"""
Model Predictive Control - CasADi interface
Adapted from Helge-André Langåker work on GP-MPC
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import casadi as ca
import casadi.tools as ctools
import pdb


class MPC(object):

    def __init__(self,
                 N: int(10), Q=None, P=None, R=None,
                 ulb=None, uub=None, xlb=None, xub=None, terminal_constraint=None,
                 solver_opts=None):
        """
        Constructor for the MPC class.

        :param model: System model
        :type model: Astrobee
        :param dynamics: Astrobee dynamics model
        :type dynamics: ca.Function
        :param N: horizion length
        :type N: int
        :param Q: state weight matrix, defaults to None
        :type Q: np.ndarray, optional
        :param P: terminal state weight matrix, defaults to None
        :type P: np.ndarray, optional
        :param R: control weight matrix, defaults to None
        :type R: np.ndarray, optional
        :param ulb: control lower bound vector, defaults to None
        :type ulb: np.ndarray, optional
        :param uub: control upper bound vector, defaults to None
        :type uub: np.ndarray, optional
        :param xlb: state lower bound vector, defaults to None
        :type xlb: np.ndarray, optional
        :param xub: state upper bound vector, defaults to None
        :type xub: [type], optional
        :param terminal_constraint: terminal constriant polytope, defaults to None
        :type terminal_constraint: Polytope, optional
        :param solver_opts: additional solver options, defaults to None.
                            solver_opts['print_time'] = False
                            solver_opts['ipopt.tol'] = 1e-8
        :type solver_opts: dictionary, optional
        """

        build_solver_time = -time.time()
        # self.dt = model.dt
        self.Nx, self.Nu = 3, 1
        self.Nt = N
        # print("Horizon steps: ", N * self.dt)
        self.dynamics = self.next_dynamics

        # Initialize variables
        self.set_cost_functions()
        self.x_sp = None

        # Cost function weights
        if P is None:
            P = np.eye(self.Nx) * 10
        if Q is None:
            Q = np.eye(self.Nx)
        if R is None:
            R = np.eye(self.Nu) * 0.01

        self.Q = ca.MX(Q)
        self.P = ca.MX(P)
        self.R = ca.MX(R)

        if xub is None:
            xub = np.full((self.Nx), np.inf)
        if xlb is None:
            xlb = np.full((self.Nx), -np.inf)
        if uub is None:
            uub = np.full((self.Nu), np.inf)
        if ulb is None:
            ulb = np.full((self.Nu), -np.inf)

        # Starting state parameters - add slack here
        x0 = ca.MX.sym('x0', self.Nx)
        x0_ref = ca.MX.sym('x0_ref', self.Nx)
        u0 = ca.MX.sym('u0', self.Nu)
        param_s = ca.vertcat(x0, x0_ref, u0)

        # Create optimization variables
        opt_var = ctools.struct_symMX([(ctools.entry('u', shape=(self.Nu,), repeat=self.Nt),
                                        ctools.entry('x', shape=(
                                            self.Nx,), repeat=self.Nt + 1),
                                        )])
        self.opt_var = opt_var
        self.num_var = opt_var.size

        # Decision variable boundries
        self.optvar_lb = opt_var(-np.inf)
        self.optvar_ub = opt_var(np.inf)

        # Set initial values
        obj = ca.MX(0)
        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []
        con_eq.append(opt_var['x', 0] - x0)

        # Generate MPC Problem
        for t in range(self.Nt):

            # Get variables
            x_t = opt_var['x', t]
            u_t = opt_var['u', t]

            # Dynamics constraint
            x_t_next = self.dynamics(x_t, u_t)
            con_eq.append(x_t_next - opt_var['x', t + 1])

            # Input constraints
            if uub is not None:
                con_ineq.append(u_t)
                con_ineq_ub.append(uub)
                con_ineq_lb.append(np.full((self.Nu,), -ca.inf))
            if ulb is not None:
                con_ineq.append(u_t)
                con_ineq_ub.append(np.full((self.Nu,), ca.inf))
                con_ineq_lb.append(ulb)

            # State constraints
            if xub is not None:
                con_ineq.append(x_t)
                con_ineq_ub.append(xub)
                con_ineq_lb.append(np.full((self.Nx,), -ca.inf))
            # if xlb is not None:
            #     con_ineq.append(x_t)
            #     con_ineq_ub.append(np.full((self.Nx,), ca.inf))
            #     con_ineq_lb.append(xlb)

            # Objective Function / Cost Function
            obj += self.running_cost((x_t - x0_ref), self.Q, u_t, self.R)

            ############ CLF ##################
            # pdb.set_trace()
            # v = (x_t[0] - 18)**2
            # v_next = (x_t_next[0] - 18)**2
            # clf = v_next + (0.5-1) * v
            # con_ineq.append(clf)
            # con_ineq_ub.append(np.array([[0]]).T)
            # con_ineq_lb.append(np.array([[-np.inf]]).T)
            # pdb.set_trace()
            delta_star = 1.8 * x_t[0] + 0.5 * \
                ((0.3 * x_t[0] ** 2 - 0.3 * x_t[1] ** 2) /
                    (0.3 * 0.3 * 9.81))
            delta_star_next = 1.8 * x_t_next[0] + 0.5 * \
                ((0.3 * x_t_next[0] ** 2 - 0.3 * x_t_next[1] ** 2) /
                    (0.3 * 0.3 * 9.81))
            h = x_t[2] + delta_star
            h_next = x_t_next[2] - delta_star_next
            # h = x_t[2] - 1.8 * x_t[0] - .5 * (x_t[0] - 0) ** 2 / (0.3 * 9.81)
            # h_next = x_t_next[2] - 1.8 * \
            #     x_t_next[0] - .5 * (x_t_next[0] - 0) ** 2 / (0.3 * 9.81)
            cbf = h_next + (1-1) * h
            con_ineq.append(cbf)
            con_ineq_ub.append(np.array([[np.inf]]).T)
            con_ineq_lb.append(np.array([[0]]).T)
            # pdb.set_trace()
        # Terminal Cost
        obj += self.terminal_cost(opt_var['x', self.Nt] - x0_ref, self.P)

        con_ineq.append(opt_var['x', self.Nt])
        con_ineq_ub.append(np.array([[20, 15, np.inf]]).T)
        con_ineq_lb.append(np.array([[0.2, 0, 7]]).T)
        # Terminal contraint
        if terminal_constraint is not None:
            # Should be a polytope
            H_N = terminal_constraint.A
            if H_N.shape[1] != self.Nx:
                print("Terminal constraint with invalid dimensions.")
                exit()

            H_b = terminal_constraint.b
            con_ineq.append(H_N @ opt_var['x', self.Nt])
            con_ineq_lb.append(-ca.inf * ca.DM.ones(H_N.shape[0], 1))
            con_ineq_ub.append(H_b)

        # Equality constraints bounds are 0 (they are equality constraints),
        # -> Refer to CasADi documentation
        num_eq_con = ca.vertcat(*con_eq).size1()
        num_ineq_con = ca.vertcat(*con_ineq).size1()
        con_eq_lb = np.zeros((num_eq_con,))
        con_eq_ub = np.zeros((num_eq_con,))

        # Set constraints
        con = ca.vertcat(*con_eq, *con_ineq)
        self.con_lb = ca.vertcat(con_eq_lb, *con_ineq_lb)
        self.con_ub = ca.vertcat(con_eq_ub, *con_ineq_ub)

        # Build NLP Solver (can also solve QP)
        nlp = dict(x=opt_var, f=obj, g=con, p=param_s)
        options = {
            'ipopt.print_level': 0,
            'print_time': False,
            'verbose': False,
            'expand': True
        }
        if solver_opts is not None:
            options.update(solver_opts)
        self.solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, options)
        # pdb.set_trace()
        # build_solver_time += time.time()
        # print('\n________________________________________')
        # print('# Time to build mpc solver: %f sec' % build_solver_time)
        # print('# Number of variables: %d' % self.num_var)
        # print('# Number of equality constraints: %d' % num_eq_con)
        # print('# Number of inequality constraints: %d' % num_ineq_con)
        # print('----------------------------------------')
        pass

    def set_cost_functions(self):
        """
        Helper method to create CasADi functions for the MPC cost objective.
        """
        # Create functions and function variables for calculating the cost
        Q = ca.MX.sym('Q', self.Nx, self.Nx)
        R = ca.MX.sym('R', self.Nu, self.Nu)
        P = ca.MX.sym('P', self.Nx, self.Nx)

        x = ca.MX.sym('x', self.Nx)
        u = ca.MX.sym('u', self.Nu)

        # Instantiate function
        self.running_cost = ca.Function('Jstage', [x, Q, u, R],
                                        [x.T @ Q @ x + u.T @ R @ u])

        self.terminal_cost = ca.Function('Jtogo', [x, P],
                                         [x.T @ P @ x])

    def solve_mpc(self, x0, u0=None):
        """
        Solve the optimal control problem

        :param x0: starting state
        :type x0: np.ndarray
        :param u0: optimal control guess, defaults to None
        :type u0: np.ndarray, optional
        :return: predicted optimal states and optimal control inputs
        :rtype: ca.DM
        """

        # Initial state
        if u0 is None:
            u0 = np.zeros(self.Nu)
        if self.x_sp is None:
            self.x_sp = np.zeros(self.Nx)

        # Initialize variables
        self.optvar_x0 = np.full((1, self.Nx), x0.T)

        # Initial guess of the warm start variables
        self.optvar_init = self.opt_var(0)
        self.optvar_init['x', 0] = self.optvar_x0[0]

        # print('\nSolving MPC with %d step horizon' % self.Nt)
        # solve_time = -time.time()

        param = ca.vertcat(x0, self.x_sp, u0)
        args = dict(x0=self.optvar_init,
                    lbx=self.optvar_lb,
                    ubx=self.optvar_ub,
                    lbg=self.con_lb,
                    ubg=self.con_ub,
                    p=param)
        # pdb.set_trace()
        # Solve NLP
        sol = self.solver(**args)
        status = self.solver.stats()['return_status']
        optvar = self.opt_var(sol['x'])

        # solve_time += time.time()
        # print('\nMPC took %f seconds to solve.' % (solve_time))
        # print('MPC cost: ', sol['f'])
        if status == "Infeasible_Problem_Detected":
            print("Infeasible_Problem_Detected")

        return optvar['x'], optvar['u'], status

    def mpc_controller(self, x0):
        """
        MPC controller wrapper.
        Gets first control input to apply to the system.

        :param x0: initial state
        :type x0: np.ndarray
        :return: control input
        :rtype: ca.DM
        """

        _, u_pred, status = self.solve_mpc(x0)

        return u_pred[0], status

    def set_reference(self, x_sp):
        """
        Set the controller reference state

        :param x_sp: desired reference state
        :type x_sp: np.ndarray
        """
        self.x_sp = x_sp

    def next_dynamics(self, x, u):
        dt = 0.2
        A = np.matrix([
            [1, 0, 0],
            [0, 1, 0],
            [-dt, dt, 1]])

        B = np.matrix([
            [dt],
            [0],
            [1/2 * dt ** 2]])
        x_next = A @ x + B @ u

        return x_next
