# in this script we will be performing a joint inversion of the
# activation energy and the temperature. We have used a simple model in 
# which the mu and T are related by the arrhenius equation. 
#  for that we take an initial guess for the inverse crime
# from that initial guess, we calculate the misfit between observed dynamic 
# topography and guessed one and also between surface velocities
# we also calculate the gradient and verify them using the taylor's test

# +
from gadopt import *
from gadopt.inverse import *
import numpy as np
import inspect
# Open the checkpoint file and subsequently load the mesh:
rmin, rmax = 1.22, 2.22
with CheckpointFile("../forward/Final_State.h5", "r") as forward_check:
    mesh = forward_check.load_mesh("firedrake_default_extruded")
    mesh.cartesian = False
    bottom_id, top_id = "bottom", "top"    
    T_obs = forward_check.load_function(mesh, "Temperature")
    dtopo_obs = forward_check.load_function(mesh, "Observed DT")
    mu_observed = forward_check.load_function(mesh, "Observed Viscosity")
    T_av = forward_check.load_function(mesh, "Average Temperature")
    mu_av = forward_check.load_function(mesh, "Average Viscosity")    
    u_obs = forward_check.load_function(mesh, "Observed Velocity")

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1) #scalar space for functions
Z = MixedFunctionSpace([V, W])  # Mixed function space.
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

# define the observed activation energy
Ea_obs = Function(R, name="Observed Activation").assign(ln(100))
mesh_func = Function(R, name="Mesh Function").assign(1.0) # to normalise terms

tape = get_working_tape()
tape.clear_tape()
print(tape.get_blocks())

z = Function(Z)  # A field over the mixed function space Z
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

# We next specify the important constants for this problem, and set up the approximation.
Ra = Function(R, name="Rayleigh Number").assign(5e4)
approximation = BoussinesqApproximation(Ra)

# timestepping parameters
time = 0.0  # Initial time
delta_t = Constant(5e-6)  # Initial time-step
timesteps = 10  # Maximum number of timesteps
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)

# We also set up a dynamic topography field for visualisation:
dtopo = Function(W, name="Dynamic_Topography")
deltarho_g = Constant(1e3) #for scaling
dtopo_error = Function(W, name="Dynamic_Topography_Error")


Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)

Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

stokes_bcs = {
    bottom_id: {'un': 0},
    top_id: {'un': 0},
}


temp_bcs = {
    bottom_id: {'T': 1.0},
    top_id: {'T': 0.0},
}

# +
stokes_solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# now we will invert for the activation energy and the temperature as our controls
# we will start by defining the control for the activation energy
Ea_control = Function(R, name="Activation Control").assign(4.5)
control1 = Control(Ea_control)
Ea = Function(R, name="Activation Energy").project(Ea_control)

# now define the control for the temperature
# boundary conditions for the temperature
T0_bcs = [DirichletBC(Q1, 0., top_id), DirichletBC(Q1, 1., bottom_id)]
T_average = Function(Q1, name="Average Temperature").project(T_av, bcs=T0_bcs)
T_control = Function(Q1, name="Temperature Control").project(T_average)
control2 = Control(T_control)
T0 = Function(Q1, name="Initial Guess Temperature").project(T_control, bcs=T0_bcs)

T_bcs = [DirichletBC(Q, 0., top_id), DirichletBC(Q, 1., bottom_id)]
T = Function(Q, name = "Temp").project(T0, bcs= T_bcs)

# define the control for the reduced functional
control = [control1, control2]

# now we will define the viscosity field which is dependent on the activation energy and the temperature
# boundary conditions for the viscosity
mu_bcs = [DirichletBC(Q1, 0.01, bottom_id), DirichletBC(Q1, 1., top_id)]
mu= Function(Q1, name="Viscosity").project((exp (-Ea * T)), bcs=mu_bcs)

# define the stokes solver
stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu, solver_parameters=stokes_solver_parameters,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)

# define the energy solver
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)

# define the surface force solver
surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)

# solve the energy and stokes equations
energy_solver.solve()
stokes_solver.solve()

# Surface force:
surface_force = surface_force_solver.solve()

# update fields
dtopo.interpolate((surface_force/deltarho_g))
dtopo_error.interpolate(dtopo_obs - dtopo)

dt_misfit = assemble((dtopo - dtopo_obs)**2 * ds_t)
u_misfit = assemble(dot(u - u_obs, u - u_obs) * ds_t)
mu_misfit = assemble((mu - mu_observed)**2 * dx)
T_misfit = assemble((T - T_obs)**2 * dx)
Ea_misfit = assemble((Ea - Ea_obs)**2 * dx)

# print the misfits
log(f"Dynamic Topography Misfit: {dt_misfit}")
log(f"Velocity Misfit: {u_misfit}")
log(f"Viscosity Misfit: {mu_misfit}")
log(f"Temperature Misfit: {T_misfit}")
log(f"Activation Energy Misfit: {Ea_misfit}")

# Define the objective functional

# define the weights for the individual terms
alpha_dt = 1.0
alpha_u = 1.0
alpha_s = 0.001
alpha_d = 0.1
alpha_r = 0.0 # no regularisation term

# Define the component terms of the overall objective functional
# surface misfits
norm_dt = assemble(dtopo_obs**2 * ds_t)
norm_u = assemble(dot(u_obs, u_obs) * ds_t)
topography_misfit = alpha_dt * dt_misfit 
velocity_misfit = alpha_u * norm_dt * (u_misfit / norm_u)

# print the components
log(f"\n\nnorm_dt: {norm_dt}")
log(f"norm_u: {norm_u}")

# smoothing terms
smoothing = assemble(dot(grad(T0 - T_av) , grad(T0 - T_av)) * dx)
norm_smoothing = assemble(dot(grad(T_obs), grad(T_obs)) * dx)
smoother = alpha_s * norm_dt * (smoothing / norm_smoothing)

# damping terms
damping = assemble((T0 - T_av) ** 2 * dx)
norm_damping = assemble(T_av**2 * dx)
damper = alpha_d * norm_dt * (damping / norm_damping)

# regularisation term (Ea)
regularisation = assemble((Ea_obs) ** 2 * dx)
norm_regularisation = assemble(mesh_func **2 * dx)
regulariser = alpha_r * norm_dt * (regularisation / norm_regularisation)

objective = (topography_misfit + velocity_misfit + smoother + damper + regulariser)

# print individual terms
log(f"\n\nTopography Term: {topography_misfit }")
log(f"Velocity Term: {velocity_misfit }")
log(f"Smoothing Term: {smoother}")
log(f"Damping Term: {damper}")
log(f"Regularisation Term: {regulariser }")
log(f"Objective: {objective}")

# Using the definition of our objective function we can define the reduced functional
reduced_functional = ReducedFunctional(objective, control)
log(f"\n\nReduced functional: {reduced_functional([Ea_control, T_control])}")

# Having the reduced functional one can easily compute the derivative
derivative_tuple = reduced_functional.derivative(options={"riesz_representation": "L2"})
# derivative wrt to the activation energy and also project it to Q1 for visualisation
derivative_Ea = Function(Q1, name="Derivative Activation Energy").project(derivative_tuple[0])
# derivative wrt to the temperature
derivative_T = derivative_tuple[1]
derivative_T.rename("Derivative Temperature")

# field visualisation
VTKFile("field-visualisation.pvd").write(*z.subfunctions, T_obs, u_obs, T, mu_observed, mu, dtopo_obs, dtopo, derivative_Ea, derivative_T, T_av, mu_av)

#  calculatin the derivative wrt each term in the objective functional
# derivative wrt to the dynamic topography misfit
reduced_functional_dt = ReducedFunctional(topography_misfit, control)
derivative_dt_tuple = reduced_functional_dt.derivative(options={"riesz_representation": "L2"})
derivative_dt_T = derivative_dt_tuple[1]
derivative_dt_T.rename("Derivative DT misfit")

# derivative wrt to the velocity misfit
reduced_functional_u = ReducedFunctional(velocity_misfit, control)
derivative_u_tuple = reduced_functional_u.derivative(options={"riesz_representation": "L2"})
derivative_u_T = derivative_u_tuple[1]
derivative_u_T.rename("Derivative Velocity misfit")

# derivative wrt to the smoothing term
reduced_functional_s = ReducedFunctional(smoother, control)
derivative_s_tuple = reduced_functional_s.derivative(options={"riesz_representation": "L2"})
derivative_s_T = derivative_s_tuple[1]
derivative_s_T.rename("Derivative Smoothing")

# derivative wrt to the damping term
reduced_functional_d = ReducedFunctional(damper, control)
derivative_d_tuple = reduced_functional_d.derivative(options={"riesz_representation": "L2"})
derivative_d_T = derivative_d_tuple[1]
derivative_d_T.rename("Derivative Damping")

# visualise the derivatives
VTKFile("derivative-visualisation.pvd").write(derivative_dt_T, derivative_u_T, derivative_s_T, derivative_d_T)

# # Performing taylor test for both the activation energy and the temperature
# Delta_Ea = Function(Ea_control.function_space(), name="Delta_Activation_Energy")
# Delta_Ea.dat.data[:] = np.random.random(Delta_Ea.dat.data.shape) * 0.1
# Delta_T = Function(T_control.function_space(), name="Delta_Temperature")
# Delta_T.dat.data[:] = np.random.random(Delta_T.dat.data.shape) * 0.1

# # Perform the Taylor test to verify the gradients
# minconv = taylor_test(reduced_functional, [Ea_control, T_control], [Delta_Ea, Delta_T])
