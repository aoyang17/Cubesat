import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid

prob = om.Problem()
model = prob.model
model.add_subsystem('p1', om.IndepVarComp('x', 50.0), promotes=['*'])
model.add_subsystem('p2', om.IndepVarComp('y', 50.0), promotes=['*'])
model.add_subsystem('comp', Paraboloid(), promotes=['*'])
model.add_subsystem('con', om.ExecComp('c = x - y'), promotes=['*'])

prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=False)

model.add_design_var('x', lower=-50.0, upper=50.0)
model.add_design_var('y', lower=-50.0, upper=50.0)
model.add_objective('f_xy')
model.add_constraint('c', lower=15.0)

filename = "cases.sql"
recorder = om.SqliteRecorder(filename)

prob.driver.add_recorder(recorder)
prob.driver.recording_options['record_desvars'] = True
prob.driver.recording_options['includes'] = []
prob.driver.recording_options['excludes'] = ['y']

prob.set_solver_print(0)
prob.setup()
prob.run_driver()
prob.cleanup()

# First case with record_desvars = True and includes = []
cr = om.CaseReader(filename)
case = cr.get_case(-1)

print(sorted(case.outputs.keys()))