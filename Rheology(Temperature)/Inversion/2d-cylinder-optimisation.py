# In this code we will perform the actual inversionusing th Lin-More optimiser
# 

from gadopt import *
from gadopt.inverse import *
import numpy as np
import inspect
# Open the checkpoint file and subsequently load the mesh and the necessary fields from the forward problem
rmin, rmax = 1.22, 2.22
with CheckpointFile("../Forward/Final_State.h5", "r") as forward_check:
    mesh = forward_check.load_mesh("firedrake_default_extruded")
    mesh.cartesian = False
    bottom_id, top_id = "bottom", "top"    
    T = forward_check.load_function(mesh, "Temperature")
    dtopo_obs = forward_check.load_function(mesh, "Observed DT")
    mu_observed = forward_check.load_function(mesh, "Observed Viscosity")
    T_average = forward_check.load_function(mesh, "Average Temperature")
    mu_av = forward_check.load_function(mesh, "Average Viscosity")    
    u_obs = forward_check.load_function(mesh, "Observed Velocity")

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1) #scalar space for functions
Z = MixedFunctionSpace([V, W])  # Mixed function space.

tape = get_working_tape()
tape.clear_tape()
print(tape.get_blocks())

z = Function(Z)  # A field over the mixed function space Z
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

# We next specify the important constants for this problem, and set up the approximation.
Ra = Constant(5e4)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

time = 0.0  # Initial time
delta_t = Constant(5e-6)  # Initial time-step
timesteps = 10  # Maximum number of timesteps
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)


# For our inverse study we need to determine the viscosity,  mu_control.
# This will be the parameter that we pretend we don't know and
# we try to invert for. So let's assume that mu is the layer average of the forward model, to start.

# control bcs
mu_bc_base = DirichletBC(Q1, 0.01, bottom_id)
mu_bc_top = DirichletBC(Q1, 1., top_id)
# Combine the boundary conditions:
mu_bcs = [mu_bc_base, mu_bc_top]

# define the control field on which we would perform the inversion
mu_average = Function(Q1, name="Average_Viscosity").project(mu_av, bcs=mu_bcs)
mu_control = Function(Q1, name="Viscosity_Control").project(mu_average)
control = Control(mu_control)

# And apply through projection of mu_average:
mu0 = Function(Q1, name="guess viscosity").project(mu_control, bcs=mu_bcs)
mu = Function(Q1, name="Viscosity").project(mu0)

# set up a mu_error field for visualisation:
mu_error = Function(Q1, name="Viscosity_Error").interpolate(mu - mu_observed)

# We also set up a dynamic topography field for visualisation:
dt_obs = Function(W, name="Observed Dynamic Topography").project(dtopo_obs)
dtopo = Function(W, name="Dynamic Topography")
deltarho_g = Constant(1e3) #for scaling
dtopo_error = Function(W, name="Dynamic_Topography_Error")

# +
# Now we will solve the Stokes system in order to calculate the model dynamic topography
# and velocity from the model viscosity field. We will then compare this to the observed 
# dynamic topography

# Define the nullspaces for the Stokes system:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)

Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

# Free-slip boundary conditions for the stokes system
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
# setting up the stokes solver
# T is the actual temperature field
# mu is the guessed viscosity field
stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu, solver_parameters=stokes_solver_parameters,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)
#                             near_nullspace=Z_near_nullspace)
 
# Define the surface force solver in order to calculate the normal stress at the surface
surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)
    
# Update the stokes solver
stokes_solver.solve()
    
# Surface force:
surface_force = surface_force_solver.solve()
    
# Update the dynamic topography and scale it by deltarho_g
dtopo.interpolate((surface_force/deltarho_g))
dtopo_error.interpolate(dtopo_obs - dtopo)

# Now at this point, we define our misfits; the difference between model and `observation`
dt_misfit = assemble ((dtopo - dtopo_obs) ** 2 * ds_t)
u_misfit = assemble(dot(u - u_obs, u - u_obs) * ds_t)
mu_misfit = assemble(0.5 * (mu - mu_observed) ** 2 * dx)

# print the misfits
log(f"DT misfit: {dt_misfit}")
log(f"U misfit: {u_misfit}")
log(f"Mu misfit: {mu_misfit}")

# define the weights for the terms within the objective function
alpha_dt = 1.0
alpha_u = 10.0
alpha_s = 0.1
alpha_d = 0.5

# Define the component terms of the overall objective functional
# surface misfits
norm_dt = assemble(dtopo_obs**2 * ds_t)
norm_u = assemble(dot(u_obs, u_obs) * ds_t)
norm_mu = assemble(mu_observed**2 * dx)
topography_misfit =  (dt_misfit )
velocity_misfit = (u_misfit  / norm_u) * norm_dt
# smoothing
smoothing = assemble(dot(grad(mu0 - mu_average ), grad(mu0 -mu_average)) * dx)
norm_smoothing = assemble(dot(grad(mu_observed), grad(mu_observed)) * dx)
smoother = (norm_dt * (smoothing / norm_smoothing))
# damping
damping = assemble((mu0 - mu_average) ** 2 * dx)
norm_damping = assemble(mu_average**2 * dx)
damper = (norm_dt * (damping / norm_damping))

# equation for the objective function
objective = alpha_dt * topography_misfit + alpha_u * velocity_misfit + alpha_s * smoother + alpha_d * damper

#  print all the components of the objective function
log(f"\n\nTopography misfit: {alpha_dt * topography_misfit}")
log(f"Velocity misfit: {alpha_u * velocity_misfit}")
log(f"Smoother: {alpha_s * smoother}")
log(f"Damper: {alpha_d * damper}")
log(f"Objective: {objective}\n\n")

# visualise different fields
VTKFile("fields-visualisation.pvd").write(dtopo, dt_obs, u_obs, mu, mu_observed, z.subfunctions[0],  T, mu_average, mu_error)

#  now we calculate the gradients wrt each of the components of the objective function
# 1. topography misfit
reduced_functional1 = ReducedFunctional(topography_misfit, control)


# ------------------------------inversion-------------------------

# Now perform inversion:
solution_pvd = VTKFile("solutions.pvd")

# objective function array for each set of weights
objective_array = []

# terminal viscosity misfit for each set of weights
viscosity_misfit_array = []

# dynamic topography misfit for each set of weights
dt_misfit_array = []

# surface velocity misfit for each set of weights
u_misfit_array = []

def callback():
    solution_pvd.write(mu_control.block_variable.checkpoint)

    #get the value of the terminal viscosity misfit
    mu_inv = Function(Q1, name="invViscosity").project(mu_control.block_variable.checkpoint)

    #get the value of objective function at each iteration
    current_objective = reduced_functional(mu_inv)
    log(f"Objective function: {current_objective}")
    #append the value of the objective function to the array
    objective_array.append(current_objective)

    # solving the stokes equation with the final solution
    z_inv = Function(Z)  # A field over the mixed function space Z
    u_inv, p_inv = split(z_inv) # Returns symbolic UFL expression for u and p
    z_inv.subfunctions[0].rename("invVelocity")
    z_inv.subfunctions[1].rename("invPressure")

    # Update the stokes solver
    stokes_solver = StokesSolver(z_inv, T, approximation, 
                                         bcs=stokes_bcs, mu=mu_inv, 
                                         solver_parameters=stokes_solver_parameters,
                                            nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)
    
    # Define the surface force solver
    surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)

    # calculate the dynamic topography
    stokes_solver.solve()
    surface_force_inv = surface_force_solver.solve()
    dtopo_inv = Function(W, name="Dynamic_Topography").interpolate(surface_force_inv/deltarho_g)
    
    # calculate the dynamic topography misfit
    dt_inv_misfit = assemble((dtopo_obs - dtopo_inv) ** 2 * ds_t)
    # normalise the misfit
    dt_inv_misfit = (dt_inv_misfit )
    log(f"Dynamic Topography Misfit: {dt_inv_misfit}")
    # append the value of the dynamic topography misfit to the array
    dt_misfit_array.append(dt_inv_misfit)

    # calculate the surface velocity misfit
    u_inv_misfit = assemble(dot(u_obs - z_inv.subfunctions[0], 
                                u_obs - z_inv.subfunctions[0]) * ds_t)
    # normalise the misfit
    u_inv_misfit = (u_inv_misfit / norm_u ) * norm_dt
    log(f"Surface Velocity Misfit: {u_inv_misfit}")
    # append the value of the surface velocity misfit to the array
    u_misfit_array.append(u_inv_misfit)

    # calculate the terminal viscosity misfit
    mu_inv_misfit = assemble(
        (mu_inv - mu_observed) ** 2 * dx
    )
    # normalise the misfit
    mu_inv_misfit = (mu_inv_misfit / norm_mu) * norm_dt
    log(f"Terminal Viscosity Misfit: {mu_inv_misfit}")
    # append the value of the terminal viscosity misfit to the array
    viscosity_misfit_array.append(mu_inv_misfit)




mu_lb = Function(mu_control.function_space(), name="Lower_bound_mu").assign(1e-2)
mu_ub = Function(mu_control.function_space(), name="Upper_bound_mu").assign(1.0)

minimisation_problem = MinimizationProblem(reduced_functional, bounds=(mu_lb, mu_ub))

# Adjust minimisation parameters:
minimisation_parameters["Status Test"]["Iteration Limit"] = 100

optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
)
optimiser.add_callback(callback)
optimiser.run()

# Now look at final solution:
optimiser.rol_solver.rolvector.dat[0].rename("Final_Solution")
with CheckpointFile("final_solution.h5", mode="w") as fi:
    fi.save_mesh(mesh)
    fi.save_function(optimiser.rol_solver.rolvector.dat[0])

VTKFile("final_solution.pvd").write(optimiser.rol_solver.rolvector.dat[0])

# print the arrays
log(f"Objective function array: {objective_array}")
log(f"Dynamic Topography Misfit array: {dt_misfit_array}")
log(f"Surface Velocity Misfit array: {u_misfit_array}")
log(f"Terminal Viscosity Misfit array: {viscosity_misfit_array}")
