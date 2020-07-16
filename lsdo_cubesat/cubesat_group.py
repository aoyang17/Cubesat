import numpy as np

from openmdao.api import Group, IndepVarComp, ExecComp

from lsdo_cubesat.attitude.attitude_group import AttitudeGroup
from lsdo_cubesat.propulsion.propulsion_group import PropulsionGroup
from lsdo_cubesat.aerodynamics.aerodynamics_group import AerodynamicsGroup
from lsdo_cubesat.orbit.orbit_group import OrbitGroup
from lsdo_cubesat.communication.comm_group import CommGroup
# from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp

from lsdo_utils.comps.arithmetic_comps.elementwise_max_comp import ElementwiseMaxComp


class CubesatGroup(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('cubesat')
        self.options.declare('mtx')
        self.options.declare('Ground_station')

    def setup(self):
        num_times = self.options['num_times']
        num_cp = self.options['num_cp']
        step_size = self.options['step_size']
        cubesat = self.options['cubesat']
        mtx = self.options['mtx']
        Ground_station = self.options['Ground_station']

        times = np.linspace(0., step_size * (num_times - 1), num_times)

        comp = IndepVarComp()

        comp.add_output('times', units='s', val=times)
        self.add_subsystem('inputs_comp', comp, promotes=['*'])

        group = AttitudeGroup(
            num_times=num_times,
            num_cp=num_cp,
            cubesat=cubesat,
            mtx=mtx,
        )
        self.add_subsystem('attitude_group', group, promotes=['*'])

        group = PropulsionGroup(
            num_times=num_times,
            num_cp=num_cp,
            step_size=step_size,
            cubesat=cubesat,
            mtx=mtx,
        )
        self.add_subsystem('propulsion_group', group, promotes=['*'])

        group = AerodynamicsGroup(
            num_times=num_times,
            num_cp=num_cp,
            step_size=step_size,
            cubesat=cubesat,
            mtx=mtx,
        )
        self.add_subsystem('aerodynamics_group', group, promotes=['*'])

        group = OrbitGroup(
            num_times=num_times,
            num_cp=num_cp,
            step_size=step_size,
            cubesat=cubesat,
            mtx=mtx,
        )
        self.add_subsystem('orbit_group', group, promotes=['*'])

        for Ground_station in cubesat.children:
            name = Ground_station['name']

            group = CommGroup(
                num_times=num_times,
                num_cp=num_cp,
                step_size=step_size,
                Ground_station=Ground_station,
                mtx=mtx,
            )

            # self.connect('times', '{}_comm_group.times'.format(name))

            self.add_subsystem('{}_comm_group'.format(name), group)

        # name = cubesat['name']
        shape = (1, num_times)
        rho = 20.

        # cubesat_name = cubesat['name']

        comp = ElementwiseMaxComp(shape=shape,
                                  in_names=[
                                      'UCSD_comm_group_Data',
                                      'UIUC_comm_group_Data',
                                      'Georgia_comm_group_Data',
                                      'Montana_comm_group_Data',
                                  ],
                                  out_name='KS_Data',
                                  rho=rho)
        self.add_subsystem('KS_Data_comp', comp, promotes=['*'])

        for Ground_station in cubesat.children:
            Ground_station_name = Ground_station['name']

            self.connect(
                '{}_comm_group.Data'.format(Ground_station_name),
                '{}_comm_group_Data'.format(Ground_station_name),
            )

        comp = ExecComp(
            'KS_total_Data = KS_Data[-1] - KS_Data[0]',
            KS_Data=np.empty(num_times),
        )
        self.add_subsystem('KS_total_Data_comp', comp, promotes=['*'])