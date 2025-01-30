# in this script we will be performing a joint inversion of the
# activation energy and the temperature. We have used a simple model in 
# which the mu and T are related by the arrhenius equation. 

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


# define the surface force solver
surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)

# solve the stokes equation
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
alpha_r = 0.0 # no regularisation term for now

# Define the component terms of the overall objective functional
# surface misfits
norm_dt = assemble(dtopo_obs**2 * ds_t)
norm_u = assemble(dot(u_obs, u_obs) * ds_t)
topography_misfit = alpha_dt * dt_misfit 
velocity_misfit = alpha_u * norm_dt * (u_misfit / norm_u)

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
log(f"Objective: {objective}")

# Using the definition of our objective function we can define the reduced functional
reduced_functional = ReducedFunctional(objective, control)
log(f"\n\nReduced functional: {reduced_functional([Ea_control, T_control])}")

# ------------------------------inversion-------------------------
# define the solution file
solution_pvd = VTKFile("solutions.pvd")
#  objective function array for convergence
objective_array = []
# terminal viscosity misfits
viscosity_misfit_array = []
# terminal temperature misfits
temperature_misfit_array = []
# terminal activation energy misfits
activation_array = []
# terminal dynamic topography misfits
dt_misfit_array = []
# terminal velocity misfits
u_misfit_array = []

def callback():
    # calculate the terminal misfits
    T_inv_misfit = assemble((T_control.block_variable.checkpoint - T_obs) ** 2 * dx)
    Ea_inv_misfit = assemble((Ea_control.block_variable.checkpoint - Ea_obs) ** 2 * dx)
    log(f"Terminal Activation Energy Misfit: {Ea_inv_misfit}")
    log(f"Terminal Temperature Misfit: {T_inv_misfit}")

    # activation value print
    Ea_terminal = assemble(Ea_control.block_variable.checkpoint * dx)/  assemble(mesh_func * dx)
    activation_array.append(Ea_terminal)
    log(f"Activation Energy: {Ea_terminal}")

    # calculate the resultant viscosity field
    mu_inv = Function(Q1, name="Viscosity").project((exp(
        (-Ea_control.block_variable.checkpoint) * T_control.block_variable.checkpoint)),
                                                    bcs=mu_bcs)
    mu_inv_misfit = assemble((mu_inv - mu_observed) ** 2 * dx)
    log(f"Terminal Viscosity Misfit: {mu_inv_misfit}")
    viscosity_misfit_array.append(mu_inv_misfit)

    # print the current value of the reduced functional
    current_objective = reduced_functional([Ea_control.block_variable.checkpoint, T_control.block_variable.checkpoint])
    log(f"Reduced functional: {current_objective}")
    # append the reduced functional to the array
    objective_array.append(current_objective)
    temperature_misfit_array.append(T_inv_misfit)

    # calculate the resultant dynamic topography field
    #  define z2
    z_inv = Function(Z)
    u_inv, p_inv = split(z_inv)
    z_inv.subfunctions[0].rename("Inverse Velocity")
    z_inv.subfunctions[1].rename("Inverse Pressure")

    # define stokes solver
    stokes_solver_inv = StokesSolver(z_inv, T_control.block_variable.checkpoint, approximation, bcs=stokes_bcs, mu=mu_inv,
                                        solver_parameters=stokes_solver_parameters,
                                        nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)
    surface_force_solver_inv = BoundaryNormalStressSolver(stokes_solver_inv, top_id)
    stokes_solver_inv.solve()
    surface_force_inv = surface_force_solver_inv.solve()
    dtopo_inv = Function(W, name="Dynamic_Topography").interpolate((surface_force_inv / deltarho_g))
    dt_misfit_inv = assemble((dtopo_inv - dtopo_obs) ** 2 * ds_t)
    log(f"Dynamic Topography Misfit: {dt_misfit_inv}")
    dt_misfit_array.append(dt_misfit_inv)

    # update the velocity field
    u_inv = z_inv.subfunctions[0]
    u_inv_misfit = assemble(dot(u_inv - u_obs, u_inv - u_obs) * ds_t)
    log(f"Velocity Misfit: {norm_dt * (u_inv_misfit / norm_u)}")
    u_misfit_array.append(u_inv_misfit)

    # write the solution in the solution file
    solution_pvd.write(T_control.block_variable.checkpoint, mu_inv, dtopo_inv, u_inv, T_obs, mu_observed, dtopo_obs, u_obs)

#  define the bounds for the controls
Ea_lb = Function(Ea_control.function_space(), name="Lower_bound_Activation").assign(0.0)
Ea_ub = Function(Ea_control.function_space(), name="Upper_bound_Activation").assign(10.0)
T_lb = Function(T_control.function_space(), name="Lower_bound_Temperature").assign(0.0)
T_ub = Function(T_control.function_space(), name="Upper_bound_Temperature").assign(1.0)

# minimisation problem
minimisation_problem = MinimizationProblem(reduced_functional, bounds=([Ea_lb, Ea_ub], [T_lb, T_ub]))

# Adjust minimisation parameters:
minimisation_parameters["Status Test"]["Iteration Limit"] = 100

# define the optimiser
optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
)
optimiser.add_callback(callback)

# run the optimiser
optimiser.run()

# Now look at final solution:
Ea_inverted = Function(Q1, name="Final_Activation_Energy").project(optimiser.rol_solver.rolvector.dat[0])
T_inverted = Function(Q1, name="Final_Solution").project(optimiser.rol_solver.rolvector.dat[1])
mu_inverted = Function(Q1, name="Final_Viscosity").project((exp((-Ea_inverted * T_inverted))))


# save the final solution
with CheckpointFile("final_solution.h5", mode="w") as fi:
    fi.save_mesh(mesh)
    fi.save_function(optimiser.rol_solver.rolvector.dat[0])
    fi.save_function(optimiser.rol_solver.rolvector.dat[1])

VTKFile("final_solution.pvd").write(T_inverted, Ea_inverted, mu_inverted)

# save the arrays as numpy arrays for later plotting
np.save("objective_array.npy", objective_array)
np.save("viscosity_misfit_array.npy", viscosity_misfit_array)
np.save("temperature_misfit_array.npy", temperature_misfit_array)
np.save("activation_misfit_array.npy", activation_array)
np.save("dt_misfit_array.npy", dt_misfit_array)
np.save("u_misfit_array.npy", u_misfit_array)
