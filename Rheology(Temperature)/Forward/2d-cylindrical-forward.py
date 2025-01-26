# in this code we will generate the fields for the forward problem
#  we will get the observed viscosity, temperature, velocity and dynamic topography
#  these fields would be used on later in the inversion process to valicate the results
from gadopt import *

# +
# Loading the mesh and final state from the initial run
rmin, rmax = 1.22, 2.22
with CheckpointFile("../Initial-Spin/Final_State.h5", "r") as forward_check:
    mesh = forward_check.load_mesh("firedrake_default_extruded")
    mesh.cartesian = False
    bottom_id, top_id = "bottom", "top"    
    T = forward_check.load_function(mesh, "Temperature")
    z = forward_check.load_function(mesh, "Stokes")

# defining the function spaces
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar) and stresses
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1) #scalar space for viscosity
Z = MixedFunctionSpace([V, W])  # Mixed function space.

u, p = split(z)  # Returns symbolic UFL expression for u and p
# -

T = Function(Q, name="Temperature").interpolate(T)


# We next specify the important constants for this problem, and set up the approximation.
Ra = Constant(5e4)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

time = 0.0  # Initial time
delta_t = Constant(5e-6)  # Initial time-step
timesteps = 10  # Maximum number of timesteps
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)

X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2)

# We next evaluate the viscosity, which depends on temperature:
mu = exp(-ln(100) * T)
# And create a field for visualisation:
mu_field = Function(Q1, name="Viscosity")

# We also set up a dynamic topography field for visualisation:
dtopo = Function(W, name="Dynamic_Topography")
deltarho_g = Constant(1e3) #delta rho = 100, g = 10

# Define the nullspaces for the Stokes system:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)

Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

# +
# We next set up the boundary conditions for the Stokes system:
 # These are the free-slip boundary conditions for the velocity field.
stokes_bcs = {
    bottom_id: {'un': 0},
    top_id: {'un': 0},
}

# We also set up the boundary conditions for the temperature field:
# The temperature at the CMB is set to maximum and at the surface to minimum.
temp_bcs = {
    bottom_id: {'T': 1.0},
    top_id: {'T': 0.0},
}
# -

# +
# We next set up the output file to write the results to:
output_file = VTKFile("output.pvd")
output_frequency = 1

plog = ParameterLog('params.log', mesh)
plog.log_str("timestep time dt maxchange u_rms nu_base nu_top energy avg_t T_min T_max")

gd = GeodynamicalDiagnostics(z, T, bottom_id, top_id, quad_degree=6)

u, p = z.subfunctions
u.rename("Velocity")
p.rename("Pressure")

# initialising the checkpoint file
checkpoint_file = CheckpointFile("Checkpoint_State.h5", "w")
checkpoint_file.save_mesh(mesh)
# -

# +
# We next set up the solvers for the energy and Stokes systems:
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, mu=mu,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             near_nullspace=Z_near_nullspace)

# We also set up the surface force solver in order to calculate the normal stress at the surface:
surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)
surface_force = surface_force_solver.solve()
# -

# Set up aspects for layer averaging:
T_average = Function(Q1, name="Average_Temperature")
mu_average = Function(Q1, name="Average_Viscosity")

# Prepare for layer average calculations:
layer_averager = LayerAveraging(mesh, quad_degree=6)

# We now initiate the time loop for 10 timsteps:

for timestep in range(0, timesteps):

    # Interpolate the viscosity
    mu_field.interpolate(mu)    
    # Calculate the layer averages for the viscosity and temperature fields:
    layer_averager.extrapolate_layer_average(T_average, layer_averager.get_layer_average(T))
    layer_averager.extrapolate_layer_average(mu_average, layer_averager.get_layer_average(mu_field))
    
    # Write output:
    if timestep % output_frequency == 0:        
        dtopo.interpolate((surface_force / deltarho_g))
        output_file.write(*z.subfunctions, T, T_average, mu_field, mu_average, dtopo)

    if timestep != 0:
        dt = t_adapt.update_timestep()
    else:
        dt = float(delta_t)
    time += dt

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Surface force:
    surface_force = surface_force_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Compute diagnostics:
    f_ratio = rmin/rmax
    top_scaling = 1.3290170684486309  # log(f_ratio) / (1.- f_ratio)
    bot_scaling = 0.7303607313096079  # (f_ratio * log(f_ratio)) / (1.- f_ratio)
    nusselt_number_top = gd.Nu_top() * top_scaling
    nusselt_number_base = gd.Nu_bottom() * bot_scaling
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))

    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(delta_t)} {maxchange} {gd.u_rms()} "
                 f"{nusselt_number_base} {nusselt_number_top} "
                 f"{energy_conservation} {gd.T_avg()} {gd.T_min()} {gd.T_max()} ")
    
    # # Storing velocity to be used in the objective F
    # checkpoint_file.save_function(u, name="Velocity", idx=timestep)


plog.close()

#  save all the fields to the final checkpoint file 
#  these fields would be used in the inverssion file
with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(T, name="Temperature")
    final_checkpoint.save_function(z, name="Stokes")
    final_checkpoint.save_function(dtopo, name="Observed DT")
    final_checkpoint.save_function(mu_field, name="Observed Viscosity")
    final_checkpoint.save_function(T_average, name = "Average Temperature")
    final_checkpoint.save_function(mu_average, name = "Average Viscosity")  
    final_checkpoint.save_function(u, name = "Observed Velocity")

    

