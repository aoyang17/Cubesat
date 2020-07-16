import numpy as np
import openmdao.api as om

from openmdao.api import pyOptSparseDriver, ExecComp
import openmdao.api as om
from lsdo_viz.api import Problem

from lsdo_cubesat.api import Swarm, Cubesat, SwarmGroup
from lsdo_cubesat.communication.ground_station import Ground_station

num_times = 1501
num_cp = 300
step_size = 95 * 60 / (num_times - 1)

if 0:
    num_times = 30
    num_cp = 3
    # step_size = 50.
    step_size = 95 * 60 / (num_times - 1)

swarm = Swarm(
    num_times=num_times,
    num_cp=num_cp,
    step_size=step_size,
    cross_threshold=0.995,
)

initial_orbit_state_magnitude = np.array([0.001] * 3 + [0.001] * 3)

Cubesat_sunshade = Cubesat(
    name='sunshade',
    dry_mass=1.3,
    initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
    approx_altitude_km=500.,
    specific_impulse=47.,
    apogee_altitude=500.001,
    perigee_altitude=499.99,
)

Cubesat_optics = Cubesat(
    name='optics',
    dry_mass=1.3,
    initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
    approx_altitude_km=500.,
    specific_impulse=47.,
    perigee_altitude=500.,
    apogee_altitude=500.,
)

Cubesat_detector = Cubesat(
    name='detector',
    dry_mass=1.3,
    initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
    approx_altitude_km=500.,
    specific_impulse=47.,
    perigee_altitude=500.002,
    apogee_altitude=499.98,
)

Cubesat_sunshade.add(
    Ground_station(
        name='UCSD',
        lon=-117.1611,
        lat=32.7157,
        alt=0.4849,
    ))
Cubesat_sunshade.add(
    Ground_station(
        name='UIUC',
        lon=-88.2272,
        lat=32.8801,
        alt=0.2329,
    ))
Cubesat_sunshade.add(
    Ground_station(
        name='Georgia',
        lon=-84.3963,
        lat=33.7756,
        alt=0.2969,
    ))
Cubesat_sunshade.add(
    Ground_station(
        name='Montana',
        lon=-109.5337,
        lat=33.7756,
        alt=1.04,
    ))

Cubesat_detector.add(
    Ground_station(
        name='UCSD',
        lon=-117.1611,
        lat=32.7157,
        alt=0.4849,
    ))
Cubesat_detector.add(
    Ground_station(
        name='UIUC',
        lon=-88.2272,
        lat=32.8801,
        alt=0.2329,
    ))
Cubesat_detector.add(
    Ground_station(
        name='Georgia',
        lon=-84.3963,
        lat=33.7756,
        alt=0.2969,
    ))
Cubesat_detector.add(
    Ground_station(
        name='Montana',
        lon=-109.5337,
        lat=33.7756,
        alt=1.04,
    ))

Cubesat_optics.add(
    Ground_station(
        name='UCSD',
        lon=-117.1611,
        lat=32.7157,
        alt=0.4849,
    ))
Cubesat_optics.add(
    Ground_station(
        name='UIUC',
        lon=-88.2272,
        lat=32.8801,
        alt=0.2329,
    ))
Cubesat_optics.add(
    Ground_station(
        name='Georgia',
        lon=-84.3963,
        lat=33.7756,
        alt=0.2969,
    ))
Cubesat_optics.add(
    Ground_station(
        name='Montana',
        lon=-109.5337,
        lat=33.7756,
        alt=1.04,
    ))

swarm.add(Cubesat_sunshade)
swarm.add(Cubesat_optics)
swarm.add(Cubesat_detector)

prob = Problem()
prob.swarm = swarm

swarm_group = SwarmGroup(swarm=swarm)
prob.model.add_subsystem('swarm_group', swarm_group, promotes=['*'])

# # obj_comp = ExecComp(
# #     'obj= 0.01 * total_propellant_used- 0.001 * total_data_downloaded + 1e-4 * (0'
# #     '+ masked_normal_distance_sunshade_detector_mm_sq_sum'
# #     '+ masked_normal_distance_optics_detector_mm_sq_sum'
# #     '+ masked_distance_sunshade_optics_mm_sq_sum'
# #     '+ masked_distance_optics_detector_mm_sq_sum'
# #     '+ sunshade_cubesat_group_relative_orbit_state_sq_sum'
# #     '+ optics_cubesat_group_relative_orbit_state_sq_sum'
# #     '+ detector_cubesat_group_relative_orbit_state_sq_sum'
# #     ') / {}'.format(num_times))

obj_comp = ExecComp(
    'obj= 0.01 * total_propellant_used- 0.001 * total_KS_data_downloaded + 1e-5 * (0'
    '+ masked_normal_distance_sunshade_detector_mm_sq_sum'
    '+ masked_normal_distance_optics_detector_mm_sq_sum'
    '+ masked_distance_sunshade_optics_mm_sq_sum'
    '+ masked_distance_optics_detector_mm_sq_sum)/{}'
    '+ 0.1 * (sunshade_cubesat_group_relative_orbit_state_sq_sum'
    '+ optics_cubesat_group_relative_orbit_state_sq_sum'
    '+ detector_cubesat_group_relative_orbit_state_sq_sum'
    ') / {}'.format(num_times, num_times))

obj_comp.add_objective('obj', scaler=1.e-3)
# obj_comp.add_objective('obj')
prob.model.add_subsystem('obj_comp', obj_comp, promotes=['*'])
for cubesat_name in ['sunshade', 'optics', 'detector']:
    prob.model.connect(
        '{}_cubesat_group.relative_orbit_state_sq_sum'.format(cubesat_name),
        '{}_cubesat_group_relative_orbit_state_sq_sum'.format(cubesat_name),
    )

prob.driver = pyOptSparseDriver()
prob.driver.options['optimizer'] = 'SNOPT'
prob.driver.opt_settings['Major feasibility tolerance'] = 1e-7
prob.driver.opt_settings['Major optimality tolerance'] = 1e-7
prob.driver.opt_settings['Iterations limit'] = 500000000
prob.driver.opt_settings['Major iterations limit'] = 1000000
prob.driver.opt_settings['Minor iterations limit'] = 500000
# prob.driver.opt_settings['Iterations limit'] = 3
# prob.driver.opt_settings['Major iterations limit'] = 3
# prob.driver.opt_settings['Minor iterations limit'] = 1

# # print(prob['total_data_downloaded'])
prob.setup(check=True)
# prob.model.list_inputs()
# prob.model.list_outputs()
# prob.model.swarm_group.sunshade_cubesat_group.list_outputs(prom_name=True)

# prob.run_driver()
prob.run()
# prob.run_model()
# prob.check_partials(compact_print=True)

print(prob['optics_cubesat_group.propellant_mass'])
print(prob['detector_cubesat_group.propellant_mass'])
print(prob['sunshade_cubesat_group.propellant_mass'])
print(prob['optics_cubesat_group.Data'])
print(prob['detector_cubesat_group.Data'])
print(prob['sunshade_cubesat_group.Data'])

if 0:
    prob.model.list_outputs(prom_name=True)
    print(prob['optics_cubesat_group.times'])
    print(prob['optics_cubesat_group.propellant_mass'])
    print(prob['optics_cubesat_group.position_km'])
    print(prob['optics_cubesat_group.velocity_km_s'])
    print(prob['optics_cubesat_group.drag_unit_vec_3xn'])
    print(prob['optics_cubesat_group.velocity_km_s'] /
          prob['optics_cubesat_group.drag_unit_vec_3xn'])
    print(prob['optics_cubesat_group.radius_km'])
    print(prob['optics_cubesat_group.speed_km_s'])
    print(prob['optics_cubesat_group.position_unit_vec'])
    print(prob['optics_cubesat_group.velocity_unit_vec'])
    print(prob['sun_unit_vec'])
    print(np.sum(prob['mask_vec']))
    print(prob['distance_sunshade_optics_km'])
    print(prob['normal_distance_sunshade_detector_km'])
    print(prob['normal_distance_optics_detector_km'])
    print(prob['observation_cross_norm'])

    # print(prob['ks_masked_distance_sunshade_optics_km'])
    print(np.max(prob['masked_distance_sunshade_optics_mm']))
    # print(prob['ks_masked_distance_optics_detector_km'])
    print(np.max(prob['masked_distance_optics_detector_mm']))

    # print(prob['ks_masked_normal_distance_sunshade_detector_km'])
    print(np.max(prob['masked_normal_distance_sunshade_detector_mm']))
    # print(prob['ks_masked_normal_distance_optics_detector_km'])
    print(np.max(prob['masked_normal_distance_optics_detector_mm']))

    print(prob['sunshade_cubesat_group.ks_altitude_km'])
    print(np.min(prob['sunshade_cubesat_group.altitude_km']))

    print(prob['optics_cubesat_group.ks_altitude_km'])
    print(np.min(prob['optics_cubesat_group.altitude_km']))

    print(prob['detector_cubesat_group.ks_altitude_km'])
    print(np.min(prob['detector_cubesat_group.altitude_km']))

    print(prob['sunshade_cubesat_group.orbit_state'][0, :])
    print(prob['sunshade_cubesat_group.reference_orbit_state'][0, :])
    print(prob['sunshade_cubesat_group.relative_orbit_state'][0, :])

    print(prob['sunshade_cubesat_group.orbit_state'][3, :])
    print(prob['sunshade_cubesat_group.reference_orbit_state'][3, :])
    print(prob['sunshade_cubesat_group.relative_orbit_state'][3, :])

    if num_times < 50:
        prob.check_partials(compact_print=True)
        print(prob['mask_vec'])

if 0:
    import matplotlib.pyplot as plt

    plt.subplot(2, 2, 1)
    for ind in range(3):
        plt.plot(
            prob['sunshade_cubesat_group.times'],
            prob['sunshade_cubesat_group.position_km'][ind, :],
        )
    plt.ylabel('position')

    plt.subplot(2, 2, 2)
    for ind in range(3):
        plt.plot(
            prob['sunshade_cubesat_group.times'],
            prob['sunshade_cubesat_group.velocity_km_s'][ind, :],
        )
    plt.ylabel('velocity')

    plt.subplot(2, 2, 3)
    for ind in range(3):
        plt.plot(
            prob['sunshade_cubesat_group.times'],
            prob['sunshade_cubesat_group.relative_orbit_state'][ind, :],
        )
    plt.ylabel('relative position')

    plt.subplot(2, 2, 4)
    for ind in range(3):
        plt.plot(
            prob['sunshade_cubesat_group.times'],
            prob['sunshade_cubesat_group.relative_orbit_state'][ind + 3, :],
        )
    plt.ylabel('relative velocity')

    plt.show()

# prob.cleanup()
np.savetxt("rundata/optics_propellant.csv",
           prob['optics_cubesat_group.propellant_mass'],
           header="optics_cubesat_group.propellant_mass",
           delimiter=',')

np.savetxt("rundata/detector_propellant.csv",
           prob['detector_cubesat_group.propellant_mass'],
           header="detector_cubesat_group.propellant_mass",
           delimiter=',')

np.savetxt("rundata/sunshade_propellant.csv",
           prob['sunshade_cubesat_group.propellant_mass'],
           header="sunshade_cubesat_group.propellant_mass",
           delimiter=',')

np.savetxt("rundata/optics_data.csv",
           prob['optics_cubesat_group.Data'],
           header="optics_cubesat_group.Data",
           delimiter=',')

np.savetxt("rundata/detector_data.csv",
           prob['detector_cubesat_group.Data'],
           header="detector_cubesat_group.Data",
           delimiter=',')

np.savetxt("rundata/sunshade_data.csv",
           prob['sunshade_cubesat_group.Data'],
           header="sunshade_cubesat_group.Data",
           delimiter=',')

np.savetxt("rundata/optics_data_rate.csv",
           prob["optics_cubesat_group.Download_rate"],
           header="optics_data_download_rate")

np.savetxt("rundata/detector_data_rate.csv",
           prob["detector_cubesat_group.Download_rate"],
           header="detector_data_download_rate")

np.savetxt("rundata/sunshade_data_rate.csv",
           prob["sunshade_cubesat_group.Download_rate"],
           header="sunshade_data_download_rate")
