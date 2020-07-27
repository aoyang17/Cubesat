import numpy as np
import openmdao.api as om

from openmdao.api import pyOptSparseDriver, ExecComp
import openmdao.api as om
from lsdo_viz.api import Problem

from lsdo_cubesat.api import Swarm, Cubesat, SwarmGroup
from lsdo_cubesat.alignment.alignment_group import AlignmentGroup
from lsdo_utils.api import get_bspline_mtx, CrossProductComp
from lsdo_cubesat.utils.decompose_vector_group import DecomposeVectorGroup
from lsdo_cubesat.utils.dot_product_comp import DotProductComp

num_times = 1501
num_cp = 300
step_size = 95 * 60 / (num_times - 1)

initial_orbit_state_magnitude = np.array([0.001] * 3 + [0.001] * 3)

swarm = Swarm(
    num_times=num_times,
    num_cp=num_cp,
    step_size=step_size,
    cross_threshold=0.995,
)

swarm.add(
    Cubesat(
        name='optics',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
        approx_altitude_km=500.,
        specific_impulse=47.,
        perigee_altitude=500.,
        apogee_altitude=500.,
    ))

mtx = get_bspline_mtx(num_cp, num_times, order=4)

prob = Problem()
prob.swarm = swarm

alignment_group = AlignmentGroup(
    swarm=swarm,
    mtx=mtx,
)

prob.model.add_subsystem('alignment_group', alignment_group, promotes=['*'])

# observation_cross_vec = prob['observation_cross_vec']
# position_unit_vec = prob['position_unit_vec']
# sun_unit_vec = prob['sun_unit_vec']

comp = CrossProductComp(
    shape_no_3=(num_times, ),
    out_index=0,
    in1_index=0,
    in2_index=0,
    out_name='normal_cross_vec',
    in1_name='position_unit_vec',
    in2_name='velocity_unit_vec',
)
prob.model.add_subsystem('normal_cross_vec_comp', comp, promotes=['*'])

group = DecomposeVectorGroup(
    num_times=num_times,
    vec_name='normal_cross_vec',
    norm_name='normal_cross_norm',
    unit_vec_name='normal_cross_unit_vec',
)
prob.model.add_subsystem('normal_cross_decomposition_group',
                         group,
                         promotes=['*'])

comp = DotProductComp(vec_size=3,
                      length=num_times,
                      a_name='observation_cross_unit_vec',
                      b_name='normal_cross_unit_vec',
                      c_name='observation_dot',
                      a_units=None,
                      b_units=None,
                      c_units=None)

prob.model.add_subsystem('observation_dot_comp', comp, promotes=['*'])

prob.setup(check=True)
prob.run_model()
print(prob['observation_dot'].shape)
print(prob['observation_cross_unit_vec'].shape)
print(prob['observation_cross_norm'].shape)
# print(prob['sun_unit_vec'].shape)
