# This code will provide us with the initial conditions for the 2D cylindrical model.
# This code is an advanced version of the previous code in which we had only temperature 
# dependant viscosity. Within temperature dependance as well we had a single Ea value.
# But in this case we will have three values for the activation and will try to invert for that. 
# we will setup a case which has depth dependant viscosity and temperature dependant viscosity
# later on we will try to invert for the temperature dependant viscosity
# for the activation - we have used values of 7.5, 5.5 and 3.5 for the lower mantle
# transition zone and upper mantle respectively.

from gadopt import *
import numpy as np

# +
# set up the geometry and mesh
rmin, rmax, ncells, nlayers = 1.22, 2.22, 256, 64
rmax_earth = 6370  # Radius of Earth [km]
rmin_earth = rmax_earth - 2900  # Radius of CMB [km]
r_410_earth = rmax_earth - 410  # 410 radius [km]
r_660_earth = rmax_earth - 660  # 660 raidus [km]
r_410 = rmax - (rmax_earth - r_410_earth) / (rmax_earth - rmin_earth)
r_660 = rmax - (rmax_earth - r_660_earth) / (rmax_earth - rmin_earth)

mesh1d = CircleManifoldMesh(ncells, radius=rmin, degree=2)  # construct a circle mesh
mesh = ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type='radial')  # extrude into a cylinder
mesh.cartesian = False
bottom_id, top_id = "bottom", "top"

# define the function spaces for storing and creating the fields
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1)  # scalar space for functions
Z = MixedFunctionSpace([V, W])  # Mixed function space.
R = FunctionSpace(mesh, "R", 0 ) # real number space

# mixed function space for the velocity and pressure
z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
# -

# We next specify the important constants for this problem, and set up the approximation.
Ra = Function(R).assign(1e7)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

# time stepping parameters for the simulation
time = 0.0  # Initial time
delta_t = Constant(1e-7)  # Initial time-step
timesteps = 20000  # Maximum number of timesteps
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)
steady_state_tolerance = 1e-7  # Used to determine if solution has reached a steady state.

# We next set up the initial temperature field, which is a function of the radial coordinate.
# it triggers four equidistant plumes in the domain
X = SpatialCoordinate(mesh)
T = Function(Q, name="Temperature")
r = sqrt(X[0]**2 + X[1]**2)
T.interpolate(rmax - r + 0.02*cos(4*atan2(X[1], X[0])) * sin((r - rmin) * pi))

# +
# define the linear components of the viscosity
# now we will define the viscosity field
# we will incorporate viscosity jumps
# this is basically the depth dependant viscosity
def step_func(centre, mag, increasing=True, sharpness=50):
    return mag * (
    0.5 * (1 + tanh((1 if increasing else -1) * (r - centre) * sharpness))
    )

# define a depth-dependent viscosity
mu_lin = 1.0

# assemble the depth dependance (depth activation)
for line, step in zip(
    [5.0 * (rmax - r), 1.0, 1.0],
    [
        step_func(r_660, 30, False),
        step_func(r_410, 10, False),
        step_func(2.2, 10, True),
    ],
):
    mu_lin += line * step

#  add temperature dependance (temperature activation)
#  within the temperature dependance we have three activation energies for the different depths
# make a function with the activation energies depth dependant
#  Ea = 7.5 for the lower mantle
#  Ea = 5.5 for the transition zone
#  Ea = 3.5 for the upper mantle
def temperature_activation_func(r, sharpness=75):
    transition_660 = 0.5 * (1 + tanh((r - r_660) * sharpness))
    transition_410 = 0.5 * (1 + tanh((r - r_410) * sharpness))

    return (7.5)*(1-transition_660) + \
            (5.5)*(transition_660 - transition_410) + \
            (3.5)*(transition_410)

# define the activation function to see.
temperature_activation = Function(Q1, name="Temperature Activation").project(temperature_activation_func(r))

# calculate the viscosity field
mu = mu_lin * (exp(-temperature_activation * T))

# not keeping non-linear components of the viscosity for this case

#  mu function for visualisation
mu_function = Function(Q1, name="Viscosity")

# # test visualisation
# VTKFile("mu.pvd").write(mu_function, temperature_activation, T)


#  setup a dynamic topography field for visualisation
dtopo = Function(W, name="Dynamic Topography")

# calculate the layer average of the initial state
# Calculate the layer average of the initial state Temp
T_average = Function(Q1, name="Average Temperature")
averager_T = LayerAveraging(
    mesh, np.linspace(rmin, rmax, nlayers * 2), quad_degree=6
)
averager_T.extrapolate_layer_average(T_average, averager_T.get_layer_average(T))

# Calculate the layer average of the initial state mu
mu_average = Function(Q1, name="Average Viscosity")
averager_mu = LayerAveraging(
    mesh, np.linspace(rmin, rmax, nlayers * 2), quad_degree=6
)
averager_mu.extrapolate_layer_average(mu_average, averager_mu.get_layer_average(mu))
# -

# create a checkpoint file to save the different states
checkpoint_file  = CheckpointFile("Final Checkpoint.h5", "w")
checkpoint_file.save_mesh(mesh)
checkpoint_file.save_function(T, name="Initial Temperature")
checkpoint_file.save_function(T_average, name="Average Temperature")
checkpoint_file.save_function(mu_average, name="Average Viscosity")

# +
# setup the nullspaces
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
Z_near_nullspace = create_stokes_nullspace(Z, closed=False, 
                                           rotational=True, translations=[0, 1])

# setup the boundary conditions
# free slip velocity boundary conditions
stokes_bcs = {
    bottom_id: {"un": 0},
    top_id: {"un": 0},
}

# temperature boundary conditions
temp_bcs = {
    bottom_id: {"T": 1.0},
    top_id: {"T": 0.0},
}

# solver parameters
stokes_solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# define the solvers
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)

stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=stokes_bcs,
    mu=mu,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
    near_nullspace=Z_near_nullspace,
    solver_parameters=stokes_solver_parameters,
)

surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)

# setup output file
output_file = VTKFile("output.pvd")
output_frequency = 50

# setup the logging file
plog = ParameterLog("params.log", mesh)
plog.log_str("timestep time dt maxchange u_rms nu_base nu_top energy avg_t T_min T_max")
gd = GeodynamicalDiagnostics(z, T, bottom_id, top_id, quad_degree=6)

# +
# initiate the time loop
for timestep in range(1, timesteps+1):

    if (timestep-1) != 0:
        dt = t_adapt.update_timestep()
    else:
        dt = float(delta_t)
    time += dt

    # Solve Stokes sytem:
    stokes_solver.solve()
    
    # Temperature system:
    energy_solver.solve()

    # Surface force:
    surface_force = surface_force_solver.solve()

    # Write output:
    if (timestep-1) % output_frequency == 0:
        mu_function.interpolate(mu)
        dtopo.interpolate(surface_force)
        output_file.write(*z.subfunctions, T, mu_function, dtopo)
        # print percentage done
        print(f"{((timestep-1)/timesteps)*100:.2f}% done")

    # Compute diagnostics:
    f_ratio = rmin / rmax
    top_scaling = 1.3290170684486309  # log(f_ratio) / (1.- f_ratio)
    bot_scaling = 0.7303607313096079  # (f_ratio * log(f_ratio)) / (1.- f_ratio)
    nusselt_number_top = gd.Nu_top() * top_scaling
    nusselt_number_base = gd.Nu_bottom() * bot_scaling
    energy_conservation = abs(abs(nusselt_number_top) - abs(nusselt_number_base))

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T - energy_solver.T_old) ** 2 * dx))

    # Log diagnostics:
    plog.log_str(
        f"{timestep} {time} {float(delta_t)} {maxchange} {gd.u_rms()} "
        f"{nusselt_number_base} {nusselt_number_top} "
        f"{energy_conservation} {gd.T_avg()} {gd.T_min()} {gd.T_max()} "
    )

    # Leave if steady-state has been achieved:
    if maxchange < steady_state_tolerance:
        log("Steady-state achieved -- exiting time-step loop")
        break

# At the end of the simulation, once a steady-state has been achieved, we close our logging file
# and checkpoint steady state temperature and Stokes solution fields to disk. These can later be
# used to restart a simulation, if required.

plog.close()
checkpoint_file.save_function(T, name="Observed Temperature")
checkpoint_file.save_function(mu_function, name="Observed Viscosity")
checkpoint_file.save_function(dtopo, name="Observed DT")
checkpoint_file.save_function(z.subfunctions[0], name="Observed Velocity")
