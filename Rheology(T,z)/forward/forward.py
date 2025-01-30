# in this script we will solve the forward problem for a 2D cylinder
#  with depth dependant viscosity and temperature dependant viscosity
# We will calculate the observed dynamic topography and velocity fields
# for the given viscosity and temperature fields

# +
# import necessary libraries
from firedrake import *
from gadopt import *
import numpy as np

# +
# set up the geometry 
rmin, rmax, ncells, nlayers = 1.22, 2.22, 256, 64
rmax_earth = 6370  # Radius of Earth [km]
rmin_earth = rmax_earth - 2900  # Radius of CMB [km]
r_410_earth = rmax_earth - 410  # 410 radius [km]
r_660_earth = rmax_earth - 660  # 660 raidus [km]
r_410 = rmax - (rmax_earth - r_410_earth) / (rmax_earth - rmin_earth)
r_660 = rmax - (rmax_earth - r_660_earth) / (rmax_earth - rmin_earth)

# load the mesh and the fields from the initial spin 
with CheckpointFile("../IC/Final Checkpoint.h5", "r") as file:
    mesh = file.load_mesh("firedrake_default_extruded")
    mesh.cartesian = False
    bottom_id, top_id = "bottom", "top"  
    T = file.load_function(mesh, "Observed Temperature")
    mu = file.load_function(mesh, "Observed Viscosity")
    T_average_initial = file.load_function(mesh, "Average Temperature")
    mu_average_initial = file.load_function(mesh, "Average Viscosity")

# define the function spaces
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
# Rayleigh number
Ra = Function(R).assign(1e7)
approximation = BoussinesqApproximation(Ra)

# set up the dynamic topography field
dtopo =  Function(W, name="Dynamic Topography")
delta_rho_g = Constant(1e3) #delta rho = 100, g = 10, scaling factor

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

# stokes solver parameters - linear solver
stokes_solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# define the stokes solver and surface solver
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
# -
# surface force solver
surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)

# solve the stokes system
stokes_solver.solve()
# calculate the dynamic topography
surface_force = surface_force_solver.solve()
dtopo.project(surface_force/ delta_rho_g)

# calculate the layer averages for observed temperature and viscosity
layer_averager = LayerAveraging(mesh, quad_degree=6)
T_average_final = Function(Q1, name="Average Final Temperature")
mu_average_final = Function(Q1, name="Average Final Viscosity")
layer_averager.extrapolate_layer_average(T_average_final, layer_averager.get_layer_average(T))
layer_averager.extrapolate_layer_average(mu_average_final, layer_averager.get_layer_average(mu))

u, p = z.subfunctions
u.rename("Velocity")
p.rename("Pressure")
# scaling factor for the observed velocity

u_obs = Function(V, name="Observed Velocity").project(u )

# visualise all the fields
VTKFile("fields.pvd").write(u_obs, T, mu, dtopo, T_average_final, mu_average_final, T_average_initial, mu_average_initial)

# save the forward state to a checkpoint file
with CheckpointFile("Forward Checkpoint.h5", "w") as file:
    file.save_mesh(mesh)
    file.save_function(u_obs, name="Observed Velocity")
    file.save_function(T, name="Observed Temperature")
    file.save_function(mu, name="Observed Viscosity")
    file.save_function(dtopo, name="Observed DT")
    file.save_function(T_average_final, name="Average Final Temperature")
    file.save_function(mu_average_final, name="Average Final Viscosity")
    file.save_function(T_average_initial, name="Average Initial Temperature")
    file.save_function(mu_average_initial, name="Average Initial Viscosity")



