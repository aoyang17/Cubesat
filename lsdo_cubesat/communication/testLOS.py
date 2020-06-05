from six.moves import range
import numpy as np

from openmdao.api import ExplicitComponent


class Comm_LOS(ExplicitComponent):
    """
    Determines if the Satellite has line of sight with the ground stations.
    """

    # constants
    Re = 6378.137

    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']

        self.add_input('r_b2g_I',
                       np.zeros((3, num_times)),
                       desc='Position vector from satellite to ground station '
                       'in Earth-centered inertial frame over time')

        self.add_input('r_e2g_I',
                       np.zeros((3, num_times)),
                       desc='Position vector from earth to ground station in '
                       'Earth-centered inertial frame over time')

        self.add_output(
            'CommLOS',
            np.zeros(num_times),
            units=None,
            desc='Satellite to ground station line of sight over time')

    def compute(self, inputs, outputs):
        num_times = self.options['num_times']
        r_b2g_I = inputs['r_b2g_I']
        r_e2g_I = inputs['r_e2g_I']
        CommLOS = outputs['CommLOS']

        Rb = 100.0
        for i in range(0, num_times):
            proj = np.dot(r_b2g_I[:, i], r_e2g_I[:, i]) / self.Re

            if proj > 0:
                CommLOS[i] = 0.
            elif proj < -Rb:
                CommLOS[i] = 1.
            else:
                x = (proj - 0) / (-Rb - 0)
                CommLOS[i] = 3 * x**2 - 2 * x**3

    def compute_partials(self, inputs, partials):
        num_times = self.options['num_times']
        r_b2g_I = inputs['r_b2g_I']
        r_e2g_I = inputs['r_e2g_I']

        self.dLOS_drb = np.zeros((num_times, 3))
        self.dLOS_dre = np.zeros((num_times, 3))

        Rb = 10.0
        for i in range(0, num_times):

            proj = np.dot(r_b2g_I[:, i], r_e2g_I[:, i]) / self.Re

            if proj > 0:
                self.dLOS_drb[i, :] = 0.
                self.dLOS_dre[i, :] = 0.
            elif proj < -Rb:
                self.dLOS_drb[i, :] = 0.
                self.dLOS_dre[i, :] = 0.
            else:
                x = (proj - 0) / (-Rb - 0)
                dx_dproj = -1. / Rb
                dLOS_dx = 6 * x - 6 * x**2
                dproj_drb = r_e2g_I[:, i]
                dproj_dre = r_b2g_I[:, i]

                self.dLOS_drb[i, :] = dLOS_dx * dx_dproj * dproj_drb
                self.dLOS_dre[i, :] = dLOS_dx * dx_dproj * dproj_dre

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dCommLOS = d_outputs['CommLOS']

        if mode == 'fwd':
            for k in range(3):
                if 'r_b2g_I' in d_inputs:
                    dCommLOS += self.dLOS_drb[:, k] * d_inputs['r_b2g_I'][k, :]
                if 'r_e2g_I' in d_inputs:
                    dCommLOS += self.dLOS_dre[:, k] * d_inputs['r_e2g_I'][k, :]
        else:
            for k in range(3):
                if 'r_b2g_I' in d_inputs:
                    d_inputs['r_b2g_I'][k, :] += self.dLOS_drb[:, k] * dCommLOS
                if 'r_e2g_I' in d_inputs:
                    d_inputs['r_e2g_I'][k, :] += self.dLOS_dre[:, k] * dCommLOS


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    num_times = 4
    Re = 6378.137

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('r_b2g_I', val=-1000 * np.random.random((3, num_times)))
    comp.add_output('r_e2g_I', val=1000 * np.random.random((3, num_times)))

    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = Comm_LOS(num_times=num_times, )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup()
    print(np.sum(prob['r_b2g_I'] * prob['r_e2g_I'], axis=0) / Re)

    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)
    print(prob['CommLOS'])