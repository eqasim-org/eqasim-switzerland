"""Simulation routing must honour the current thread limit after cached prepare."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import matsim.simulation.run as simulation


class SimulationRunThreadsTest(unittest.TestCase):
    def test_global_routing_threads_are_overridden_in_both_simulators(self):
        for use_vdf in [False, True]:
            for threads in [6, 18]:
                with self.subTest(use_vdf=use_vdf, threads=threads):
                    settings = {'threads':threads, 'last_iteration':2, 'use_vdf':use_vdf,
                                'useScheduleBasedTransport':True,
                                'preventwaitingtoentertraffic':False, 'writeexperiencedplans':False}
                    class Context:
                        def config(self, key): return settings[key]
                        def stage(self, key): return 'cached_config.xml'
                        def path(self, stage=None): return '/tmp/simulation_thread_test'
                    with ExitStack() as stack:
                        for helper in ['get_dmc_parameters_args', 'get_mode_shares_calibration_args',
                                       'get_delays_args', 'get_network_calibration_args']:
                            stack.enter_context(patch.object(simulation, helper, return_value=[]))
                        stack.enter_context(patch.object(simulation.os.path, 'exists', return_value=True))
                        java = stack.enter_context(patch.object(simulation.eqasim, 'run'))
                        simulation.execute(Context())
                    arguments = java.call_args.args[2]
                    global_key = '--config:global.numberOfThreads'
                    self.assertEqual(arguments.count(global_key), 1)
                    self.assertEqual(arguments[arguments.index(global_key)+1], str(threads))
                    qsim_key = '--config:qsim.numberOfThreads'
                    self.assertEqual(arguments[arguments.index(qsim_key)+1],
                                     str(min(threads, 16 if use_vdf else 12)))


if __name__ == '__main__':
    unittest.main()
