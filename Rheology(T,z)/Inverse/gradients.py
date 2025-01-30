# in this script we perform an inversion for the temperature activation
#  the temperature activation is a depth dependant activation energy function
#  so we will be aiming to recover the depth dependant activation energy function
# for this case we assume that we have a good knowledge about the temperature field 
# and the radial depth-dependance of the viscosity
# this viscosity field is quite realistic and only lacks the non-linear components

# +
# import necessary libraries
from firedrake import *
from gadopt import *
import numpy as np
from firedrake.__future__ import interpolate
from gadopt.inverse import *

# +
# set up the geometry
rmin, rmax, ncells, nlayers = 1.22, 2.22, 256, 64
rmax_earth = 6370  # Radius of Earth [km]
rmin_earth = rmax_earth - 2900  # Radius of CMB [km]
r_410_earth = rmax_earth - 410  # 410 radius [km]
r_660_earth = rmax_earth - 660  # 660 raidus [km]
r_410 = rmax - (rmax_earth - r_410_earth) / (rmax_earth - rmin_earth)
r_660 = rmax - (rmax_earth - r_660_earth) / (rmax_earth - rmin_earth)

# load the mesh and the fields from the forward problem
with CheckpointFile("../forward/Forward Checkpoint.h5", "r") as file:
    mesh = file.load_mesh("firedrake_default_extruded")
    mesh.cartesian = False
    bottom_id, top_id = "bottom", "top"  
    T_obs = file.load_function(mesh, "Observed Temperature")
    mu_obs = file.load_function(mesh, "Observed Viscosity")
    dt_obs = file.load_function(mesh, "Observed DT")
    u_obs = file.load_function(mesh, "Observed Velocity")

# define the function spaces
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Q1 = FunctionSpace(mesh, "CG", 1)  # scalar space for functions
Z = MixedFunctionSpace([V, W])  # Mixed function space.
R = FunctionSpace(mesh, "R", 0 ) # real number space

tape = get_working_tape()
tape.clear_tape()
print(tape.get_blocks())

# mixed function space for the velocity and pressure
z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
# -

# We next specify the important constants for this problem, and set up the approximation.
Ra = Function(R, name = "Rayleigh Number").assign(1e7)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

# define the radial coordinate
X = SpatialCoordinate(mesh)
r = sqrt(X[0]**2 + X[1]**2)
# define a mesh function for averaging
mesh_func = Function(R).assign(1.0)

# define the depth dependance step function
def step_func(centre, mag, increasing=True, sharpness=50):
    return mag * (
    0.5 * (1 + tanh((1 if increasing else -1) * (r - centre) * sharpness))
    )

# activation field step function
def temperature_activation_func(r, Ea1, Ea2, Ea3, sharpness=75):
    transition_660 = 0.5 * (1 + tanh((r - r_660) * sharpness))
    transition_410 = 0.5 * (1 + tanh((r - r_410) * sharpness))

    return Ea1 * (1-transition_660) + \
            Ea2 * (transition_660 - transition_410) + \
            Ea3 * (transition_410)

# observed activation energies
Ea1_obs = Function(R, name = "Observed Activation Energy 1").assign(7.5)
Ea2_obs = Function(R, name = "Observed Activation Energy 2").assign(5.5)
Ea3_obs = Function(R, name = "Observed Activation Energy 3").assign(3.5)
Ea_obs = Function(Q1, name = "Observed Activation Energy").project(temperature_activation_func(r, Ea1_obs, Ea2_obs, Ea3_obs))

# ------------inverse crime----------------
# we pretend that we do not know the temperature activation field and we will try to recover it
#  define the controls for the inversion
Ea1 = Function(R, name = "Activation Energy 1").assign(5.0) #control 1, with initial guess
Ea2 = Function(R, name = "Activation Energy 2").assign(4.0) #control 2, with initial guess
Ea3 = Function(R, name = "Activation Energy 3").assign(5.0) #control 3, with initial guess

# define the activation energy field
Ea = Function(Q1, name = "Activation Energy").project(temperature_activation_func(r, Ea1, Ea2, Ea3))

# define the control
controls = [Control(Ea1), Control(Ea2), Control(Ea3)]

# calculate the resulting viscosity field
mu_lin = 1.0
for line, step in zip(
    [5.0 * (rmax - r), 1.0, 1.0],
    [
        step_func(r_660, 30, False),
        step_func(r_410, 10, False),
        step_func(2.2, 10, True),
    ],
):
    mu_lin += line * step

# adding the temperature dependance
mu = mu_lin * (exp(-Ea * T_obs))
# mu_eff = conditional(mu_lin > 0, mu_lin, 0)
# mu = conditional(mu_eff < 100, mu_eff, 100)
#  getting negative values of viscosity due to floating point errors ??

mu = Function(Q1, name="Viscosity").project(mu)

# mu_error function
mu_error = Function(Q1, name="Viscosity Error").assign(mu - mu_obs)

# calculate the dynamic topography from mu
dtopo = Function(W, name="Dynamic Topography") 
delta_rho_g = Constant(1e3) #delta rho = 100, g = 10, scaling factor

# define the nullspaces for the Stokes system:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)

Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

# We next set up the boundary conditions for the Stokes system:
# free-slip boundary conditions for the velocity field.
stokes_bcs = {
    bottom_id: {'un': 0},
    top_id: {'un': 0},
}
# define the stokes solver and surface solver
stokes_solver = StokesSolver(
    z,
    T_obs,
    approximation,
    bcs=stokes_bcs,
    mu=mu,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
    near_nullspace=Z_near_nullspace,
    solver_parameters={
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)
surface_force_solver = BoundaryNormalStressSolver(stokes_solver, top_id)

# solve the stokes system
stokes_solver.solve()
# calculate the dynamic topography
surface_force = surface_force_solver.solve()
dtopo.project(surface_force/ delta_rho_g)

# define u
u, p = z.subfunctions
u.rename("Velocity")
p.rename("Pressure")

# calculate the misfits
dt_misfit = assemble((dtopo - dt_obs) ** 2 * ds_t)
u_misfit = assemble(dot(u - u_obs, u - u_obs) * ds_t)
mu_misfit = assemble((mu - mu_obs) ** 2 * dx)
Ea_misfit = assemble((Ea - Ea_obs) ** 2 * dx)

# +
#  we will now set up our objective function
# define weights for the misfits
alpha_dt = 1.0
alpha_u = 1.0
alpha_r = 1.0 # regularization term

# calculate the normalisation factors
norm_dt = assemble(dt_obs ** 2 * ds_t)
norm_u = assemble(dot(u_obs, u_obs) * ds_t)
# normalised misfit
u_misfit_norm = norm_dt * u_misfit / norm_u

# define the objective function
objective = alpha_dt * dt_misfit  + alpha_u * u_misfit_norm

# # define the adjoint reduced functional
reduced_functional = ReducedFunctional(objective, controls)

# print all these terms
log(f"Dynamical Topography term: {alpha_dt * dt_misfit}")
log(f"Velocity term: {alpha_u * u_misfit_norm}")
log(f"Objective function: {objective}")
log(f"Reduced functional: {reduced_functional([Ea1, Ea2, Ea3])}")

#  visualise the fields
VTKFile("field-visualisation.pvd").write(Ea, Ea_obs, mu, mu_obs, mu_error, dtopo, dt_obs, u, u_obs, T_obs)

#----------------------calculate the gradients--------------------------
# calculate the gradients fo individual terms in the objective wrt the controls
# 1. topography misfit
reduced_functional_dt = ReducedFunctional(dt_misfit, controls)
derivative_tuple_dt = reduced_functional_dt.derivative(options={"riesz_representation": "L2"})
grad_dt_Ea1 = assemble(derivative_tuple_dt[0] * dx) / assemble(mesh_func * dx)
grad_dt_Ea2 = assemble(derivative_tuple_dt[1] * dx) / assemble(mesh_func * dx)
grad_dt_Ea3 = assemble(derivative_tuple_dt[2] * dx) / assemble(mesh_func * dx)
# print the gradients
log(f"\n\nd(dt_misfit)/dEa1: {grad_dt_Ea1}")
log(f"d(dt_misfit)/dEa2: {grad_dt_Ea2}")
log(f"d(dt_misfit)/dEa3: {grad_dt_Ea3}\n\n")

# 2. velocity misfit
reduced_functional_u = ReducedFunctional(u_misfit_norm, controls)
derivative_tuple_u = reduced_functional_u.derivative(options={"riesz_representation": "L2"})
grad_u_Ea1 = assemble(derivative_tuple_u[0] * dx) / assemble(mesh_func * dx)
grad_u_Ea2 = assemble(derivative_tuple_u[1] * dx) / assemble(mesh_func * dx)
grad_u_Ea3 = assemble(derivative_tuple_u[2] * dx) / assemble(mesh_func * dx)
# print the gradients
log(f"\n\nd(u_misfit)/dEa1: {grad_u_Ea1}")
log(f"d(u_misfit)/dEa2: {grad_u_Ea2}")
log(f"d(u_misfit)/dEa3: {grad_u_Ea3}\n\n")

# objective function
derivative_tuple = reduced_functional.derivative(options={"riesz_representation": "L2"})
grad_Ea1 = assemble(derivative_tuple[0] * dx) / assemble(mesh_func * dx)
grad_Ea2 = assemble(derivative_tuple[1] * dx) / assemble(mesh_func * dx)
grad_Ea3 = assemble(derivative_tuple[2] * dx) / assemble(mesh_func * dx)
# print the gradients
log(f"\n\nObjective function: {grad_Ea1}")
log(f"Objective function: {grad_Ea2}")
log(f"Objective function: {grad_Ea3}\n\n")

# -----------------taylor test---------------------------------
# now we will verify the gradients using the taylor test
# define the perturbation
delta_Ea1 = Function(R, name = "Perturbation1")
delta_Ea1.dat.data[:] = np.random.random(delta_Ea1.dat.data.shape)*0.1
delta_Ea2 = Function(R, name = "Perturbation2")
delta_Ea2.dat.data[:] = np.random.random(delta_Ea2.dat.data.shape)*0.1
delta_Ea3 = Function(R, name = "Perturbation3")
delta_Ea3.dat.data[:] = np.random.random(delta_Ea3.dat.data.shape)*0.1

# perform the taylor tests - topography misfit
minconv_dt = taylor_test(reduced_functional_dt, [Ea1, Ea2, Ea3], [delta_Ea1, delta_Ea2, delta_Ea3])

# perform the taylor tests - velocity misfit
minconv_u = taylor_test(reduced_functional_u, [Ea1, Ea2, Ea3], [delta_Ea1, delta_Ea2, delta_Ea3])
                                                    





