# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
The DissociationCurve is a helper class for computing, saving, and plotting full dissociation curves. When provided
with a pyscf string and list of points with which to parameterize that string (using .format()), it will use
reasonable default parameters to calculate and save the ground state energy for each point on the curve.

DissociationCurve("Li2", 'Li 0.0 0.0 0.0; Li 0.0 0.0 {0}', list(np.arange(1.5, 4.0, .05))).run()

li2_points = list(np.arange(1.5, 4.0, .05))
li2_points += list(np.arange(4.0, 5.1, .1))
li2_step = DissociationCurve("Li2", 'Li 0.0 0.0 0.0; Li 0.0 0.0 {0}', li2_points)

In addition to running, the class will:

- Run the calculation with the classical ExactEigensolver to monitor error.
- Save intermediate results to a pickle file, so the computation can be stopped and started, and results can be
printed later.
- Save each VQE run's logfile in realtime, so progress can be monitored.
- Initialize AquaChemistry dictionaries for each point, which can be modified before calling run.
- Prints results as CSV or as matplot-plotable lists.

Object Structure
 - points_to_run
 - pickle_filename
 - quantum_instance
 - vqe_results - dict of points and results
 - vqe_objects - dict of points and vqe objects
 - vqe_aqua_dicts - dict of points and chem dicts
 - vqe_dict_template
 - classical_results -
 - exact_objects -
 - exact_aqua_dicts -
 - exact_dict_template
"""

import multiprocessing
import re
import json
import logging          # to save aqua-chemistry DEBUG logs to local file
import time             # to view algorithm runtimes
import pickle           # to save/checkpoint results
import os               # to create output directory if necessary
import copy             # to create copies of aqua dicts
import numpy as np      # to extrapolate initial parameters
import pylab #TODO remove dependency

from .qiskit_chemistry import QiskitChemistry

logger = logging.getLogger(__name__)

class DissociationCurve():

    VQE_DICT_TEMPLATE = {
        'problem': {'name': 'energy',
                    'auto_substitutions': True,
                    'random_seed': 50,
                    'circuit_caching': True,
                    'skip_qobj_deepcopy': True,
                    'skip_qobj_validation': True,
                    'circuit_cache_file': None,
                    },
        'driver': {'name': 'PYSCF'},
        'PYSCF': {'unit': 'Angstrom',
                  'charge': 0,
                  'spin': 0,
                  'max_memory': None,
                  'basis': 'sto3g'},
        'algorithm': {'name': 'VQE',
                      'operator_mode': 'matrix',
                      'initial_point': None,
                      'batch_mode': False
                      },
        'operator': {'name': 'hamiltonian',
                     'transformation': 'full',
                     'freeze_core': True,
                     'orbital_reduction': [],
                     'qubit_mapping': 'parity',
                     'max_workers': multiprocessing.cpu_count(),
                     'two_qubit_reduction': True},
        'optimizer': {'disp': True, 'name': 'SLSQP', 'maxiter': 50000},
        'variational_form': {'active_occupied': None,
                             'active_unoccupied': None,
                             'depth': 1,
                             'name': 'UCCSD',
                             'num_orbitals': 4,
                             'num_particles': 2,
                             'num_time_slices': 1,
                             'qubit_mapping': 'parity',
                             'two_qubit_reduction': True},
        'backend': {'name': 'statevector_simulator', 'skip_transpiler': True, 'shots':1},
        'initial_state': {'name': 'HartreeFock',
                          'num_orbitals': 4,
                          'num_particles': 2,
                          'qubit_mapping': 'parity',
                          'two_qubit_reduction': True},
    }

    EXACT_EIGEN_DICT_TEMPLATE = {
        'driver': {'name': 'PYSCF'},
        'PYSCF': {'unit': 'Angstrom',
                  'charge': 0,
                  'spin': 0,
                  'basis': 'sto3g'},
        'algorithm': {'name': 'ExactEigensolver'},
        'operator': {'name': 'hamiltonian',
                     'transformation': 'full',
                     'freeze_core': True,
                     'orbital_reduction': [],
                     'qubit_mapping': 'parity',
                     'max_workers': multiprocessing.cpu_count(),
                     'two_qubit_reduction': True},
    }

    def __init__(self,
                 molecule_name,
                 molecule_string,
                 points_to_run,
                 point_sigfigs = 5,  # For convenience because np.arange sometimes leaves tiny imprecisions
                 compare_to_classical = True,
                 output_dir=None,
                 readonly_mode=False,
                 vqe_pickle_filename=None,
                 ee_pickle_filename=None,
                 depth = 1,
                 try_loading_params=False,
                 force_rerun_pts=[], ):
        self._molecule_name = molecule_name
        self._molecule_string = molecule_string
        self._points_to_run = [round(point, ndigits=point_sigfigs) for point in points_to_run]
        self._force_rerun_pts = [round(point, ndigits=point_sigfigs) for point in force_rerun_pts]
        self._run_classical = compare_to_classical
        self._point_sigfigs = point_sigfigs
        self._outdir = output_dir if output_dir is not None else molecule_name + "_output/"
        if not os.path.exists(self._outdir):
            os.makedirs(self._outdir)
        self._try_loading_params = try_loading_params
        self._readonly_mode = readonly_mode #Note, still saves a cache to file right now, so not totally readonly
        if not os.path.exists(self._outdir):
            os.makedirs(self._outdir)
        # if not self._readonly_mode:
        if vqe_pickle_filename : self._vqe_pickle_filename = vqe_pickle_filename
        else : self._vqe_pickle_filename = "{}_vqe.pickle".format(molecule_name)
        if ee_pickle_filename : self._ee_pickle_filename = ee_pickle_filename
        else : self._ee_pickle_filename = "{}_ee.pickle".format(molecule_name)

        # load in exact eigensolver results pickle
        self._ee_results = None
        if self._ee_pickle_filename is not None:
            try:
                self._ee_results = pickle.load(open(self._outdir + self._ee_pickle_filename, "rb"))
                print("EE results loaded successfully for points:", list(self._ee_results.keys()))
            except:
                print("EE results pickle load failed, recomputing")
        # load in vqe results pickle
        self._vqe_results = {'results': {}, 'dicts': {}}
        if self._vqe_pickle_filename is not None:
            try:
                self._vqe_results = pickle.load(open(self._outdir + self._vqe_pickle_filename, "rb"))
                print("VQE results loaded successfully for points:", list(self._vqe_results['results'].keys()))
                if 'dicts' not in self._vqe_results: self._vqe_results['dicts'] = {}
            except:
                print("VQE results pickle load failed, recomputing")

        # TODO change to create dicts in real time during run, and allow user to modify template.
        self.vqe_aqua_dict_template = copy.deepcopy(self.VQE_DICT_TEMPLATE)
        if self._run_classical:
            self.classical_aqua_dict_template = copy.deepcopy(self.EXACT_EIGEN_DICT_TEMPLATE)

        self.generate_aqua_dicts(self._points_to_run, depth=depth)

    def generate_aqua_dicts(self, points, depth=1):
        if not hasattr(self, '_dicts'): self._dicts = {}
        for point in points:
            aqua_dict = copy.deepcopy(self.VQE_DICT_TEMPLATE)
            aqua_dict['PYSCF']['atom'] = self._molecule_string.format(point)
            aqua_dict['variational_form']['depth'] = depth
            cache_file_name = self._outdir + '{0}-{1[variational_form][name]}-' \
                                             'd{1[variational_form][depth]}-{1[algorithm][operator_mode]}-cache.pickle' \
                .format(self._molecule_name, aqua_dict)
            if not self._readonly_mode:
                aqua_dict['problem']['circuit_cache_file'] = cache_file_name
            self._dicts[point] = aqua_dict

    def run(self):
        self.run_ee()
        self.run_vqe()

    def run_vqe(self, points=None):
        if not points: points = self._points_to_run
        run_points = [pt for pt in points if (pt not in self._vqe_results['results'] or pt in self._force_rerun_pts)]
        for point in run_points:
            result = self.run_vqe_pt(point)
            if point in self._ee_results:
                result['error'] = result['energy'] - self._ee_results[point]['energy']
            else:
                print("Exact solution missing for point {}, setting error to -1".format(point))
                result['error'] = -1
            result['dipole'] = result['total_dipole_moment'] / 0.393430307
            result['opt_params'] = result['algorithm_retvals']['opt_params']
            self._vqe_results['results'][point] = result
            self._vqe_results['dicts'][point] = self._dicts[point]

            self.sort_dicts()
            # checkpoint results array
            if self._vqe_pickle_filename is not None and not self._readonly_mode:
            # TODO: better to just set _vqe_pickle_filename to null if readonly and not also check if readonly here?
                pickle.dump(self._vqe_results, open(self._outdir + self._vqe_pickle_filename, 'wb'))
                print("Results object saved to " + self._outdir + self._vqe_pickle_filename)
            print('Energy: ', result['energy'])
            print('Error:', result['error'])
            self.print_results()
        # self.print_results(print_params=True)
        print('All points complete')

    def run_vqe_pt(self, point):
        if point not in self._dicts: self.generate_aqua_dicts([point])
        aqua_dict = self._dicts[point]
    # File format: N2-.6A-UCCSD-d1-SLSQP
        log_file_name = ('{0}-{1:.' + self._point_sigfigs.__str__() + 'g}A-{2[variational_form][name]}-' \
                            'd{2[variational_form][depth]}-{2[optimizer][name]}.txt') \
            .format(self._molecule_name, point, aqua_dict)
        opt_params = self.guess_params(point)
        if self._try_loading_params:
            try:
                opt_params = self.try_loading_opt_params(self._outdir+log_file_name)
            except Exception as e:
                print('Failed to load opt params from logfile')
        if opt_params is not None and len(opt_params) > 0: aqua_dict['algorithm']['initial_point'] = opt_params
        solver = QiskitChemistry()
    # Set up logging and run timer
        if not self._readonly_mode:
            loggerc = logging.getLogger('qiskit_aqua_chemistry')
            loggerc.setLevel(logging.DEBUG)
            loggera = logging.getLogger('qiskit_aqua')
            loggera.setLevel(logging.DEBUG)
            loggerq = logging.getLogger('qiskit')
            loggerq.setLevel(logging.DEBUG)
            formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            hdlr = logging.FileHandler(self._outdir + log_file_name, mode='w')
            hdlr.setFormatter(formatter)
            loggerc.addHandler(hdlr)
            loggera.addHandler(hdlr)
            loggerq.addHandler(hdlr)
            print('\nlog file: {}'.format(self._outdir + log_file_name))
        t = time.process_time()
    # run, store results, set up next iteration
        result = solver.run(aqua_dict)
        if not self._readonly_mode:
        # write out results to log file prettily
            for line in result['printable']:
                loggerc.info(line)
        # close logging and timer, return
            loggerc.removeHandler(hdlr)
            loggera.removeHandler(hdlr)
            loggerq.removeHandler(hdlr)
            hdlr.close()
        print("\nProcess time: {} mins".format((time.process_time() - t) / 60))
        return result

    #this will either run all points, or only run those missing from the results dict
    def run_ee(self, points=None):
        if not points: points = self._points_to_run
        if self._ee_results is None : self._ee_results = {}
        for point in self._points_to_run :
            if point not in self._ee_results :
                print('Exact Eigen: Processing point __\b\b{}'.format(point))
                aqua_dict = copy.deepcopy(self.EXACT_EIGEN_DICT_TEMPLATE)
                aqua_dict['PYSCF']['atom'] = self._molecule_string.format(point)
                solver = QiskitChemistry()

                t = time.process_time()
                result = solver.run(aqua_dict)
                result['dipole'] = result['total_dipole_moment'] / 0.393430307
                self._ee_results[point] = result
                print("\nProcess time: {:.2f}".format(time.process_time() - t))
                if (time.process_time() - t) > 30 and not self._readonly_mode:
                    self.sort_dicts()
                    pickle.dump(self._ee_results, open(self._outdir + self._ee_pickle_filename, 'wb'))
        if not self._readonly_mode:
            self.sort_dicts()
            pickle.dump(self._ee_results, open(self._outdir+self._ee_pickle_filename, 'wb'))
            print("Results saved to " + self._ee_pickle_filename)

    #sorts results by distance before printing
    def print_results(self, print_params=False):
        self.sort_dicts()
        distances = list(self._vqe_results['results'].keys())
        vqe_energies = [result['energy'] for result in self._vqe_results['results'].values()]
        ee_energies = [self._ee_results[pt]['energy'] for pt in list(self._vqe_results['results'].keys())
                       if pt in self._ee_results.keys()]
        errors = [self._vqe_results['results'][pt]['energy']-self._ee_results[pt]['energy'] for pt in
                  self._vqe_results['results'].keys() if pt in self._ee_results.keys()]
        print('distances =', distances)
        print('vqe_energies =', vqe_energies)
        print('ee_energies =', ee_energies)
        print('errors =', errors)

        if print_params:
            params = {pt: np.array(res['opt_params']) for (pt, res) in self._vqe_results['results'].items()}
            print("Optimal Parameters:")
            for key, value in params.items():
                print(key, end = ": ")
                print(list(value), end = ",\n")

    # if this is giving you problems on Mac OSX, run 'echo "backend: TkAgg" > ~/.matplotlib/matplotlibrc' from your
    # command prompt.
    # Requires matplotlib.
    def plot_results(self):
        distances = list(self._vqe_results['results'].keys())
        vqe_energies = [result['energy'] for result in self._vqe_results['results'].values()]
        ee_energies = [self._ee_results[pt]['energy'] for pt in list(self._vqe_results['results'].keys())
                       if pt in self._ee_results.keys()]
        pylab.plot(distances, vqe_energies, 'bo', label='VQE')
        pylab.plot(distances, ee_energies, label='Exact-Eigen')
        pylab.xlabel('Bond distance (Å)')
        pylab.ylabel('Energy (Hartrees)')
        pylab.title('{} Ground State Energy'.format(self._molecule_name))
        pylab.legend(loc='upper right')

    def sort_dicts(self):
        if self._vqe_results:
            self._vqe_results['results'] = {x[0]: x[1] for x in sorted(self._vqe_results['results'].items(), key=lambda t: t[0])}
            self._vqe_results['dicts'] = {x[0]: x[1] for x in sorted(self._vqe_results['dicts'].items(), key=lambda t: t[0])}
        if self._ee_results:
            self._ee_results = {x[0]: x[1] for x in sorted(self._ee_results.items(), key=lambda t: t[0])}

    def save_dict_changes(self):
        if not self._readonly_mode:
            self.sort_dicts()
            pickle.dump(self._vqe_results, open(self._outdir + self._vqe_pickle_filename, 'wb'))
            pickle.dump(self._ee_results, open(self._outdir + self._ee_pickle_filename, 'wb'))

    # TODO delete_results(pts)

    # TODO remove this
    def try_loading_opt_params(self, log_file_name):
        logfile = open(log_file_name, 'r')
        paramstring = None
        for line in logfile:
            if line.find('opt_params') > 0:
                paramstring = line
            elif paramstring is not None:
                paramstring += line
                if line.find(']}') > 0: break
        vals = paramstring.split('[')[1].split(']')[0].split(',')
        return  [float(i.strip()) for i in vals]

    def load_dicts_from_log_files(self, run_ee = True):
        re_strs = [r'JSON Input:',
                   r'Aqua Chemistry Input Schema:',
                   r'Algorithm returned:',
                   r'Processing complete. Final result available']
        res = []
        for re_str in re_strs:
            res.append(re.compile(re_str))
        for filename in os.listdir(self._outdir):
            if '.txt' not in filename: continue

            try:
                file_name_parts = filename.split('-')
                if not file_name_parts[0] == self._molecule_name: continue
                # pt = float(file_name_parts[1][:-1])

                logfile = open(self._outdir+filename, 'r')
                aqua_dict_str = None
                for line in logfile:
                    if re.search(re.compile(r'JSON Input:'), line):
                        aqua_dict_str = '{'
                    elif re.search(re.compile(r'Input Schema:'), line):
                        break
                    elif aqua_dict_str:
                        aqua_dict_str += line
                aqua_dict = json.loads(aqua_dict_str)
                res_dict_str = None
                for line in logfile:
                    result_line = re.search(re.compile(r'Algorithm returned:'), line)
                    if result_line:
                        res_dict_str = line[result_line.end():]
                    elif re.search(re.compile(r'Processing complete. Final result available'), line):
                        break
                    elif res_dict_str:
                        res_dict_str += line
                # if res_dict_str is None: continue
                res_dict_str = re.sub(r'\s+', '', res_dict_str)
                res_dict = eval(res_dict_str)
                printable = []
                for line in logfile:
                    if re.search(re.compile(r'=== GROUND STATE ENERGY ==='), line):
                        printable = [line]
                    elif re.search(re.compile(r'(debye)'), line):
                        printable += [line]
                        break
                    elif res_dict_str:
                        printable += [line]
                # Extract real point
                pt_ind = self._molecule_string.find(r'{0}')
                pt_str = re.split(re.compile(r'[ ;]'), aqua_dict['pyscf']['atom'][pt_ind:])[0]
                pt = np.around(float(pt_str), decimals=self._point_sigfigs)

                if pt not in self._vqe_results['dicts']:
                    self._vqe_results['dicts'][pt] = aqua_dict

                if pt not in self._vqe_results['results']:
                    self._vqe_results['results'][pt] = {}
                    self._vqe_results['results'][pt]['algorithm_retvals'] = res_dict
                    self._vqe_results['results'][pt]['printable'] = printable
                elif aqua_dict['variational_form']['depth'] > self._vqe_results['dicts'][pt]['variational_form']['depth']:
                    self._vqe_results['results'][pt]['algorithm_retvals'] = res_dict
                    self._vqe_results['results'][pt]['printable'] = printable
                    self._vqe_results['dicts'][pt] = aqua_dict
                if not self._ee_results or pt not in self._ee_results: self._points += [pt]
            except e:
                print('Failed to load results from file {}'.format(filename))
        # Run ExactEigensolver for any new points
        if run_ee: self.run_ee()

    def print_results_csv(self):
        self.sort_dicts()
        results = []
        for pt in self._vqe_results['results']:
            if pt not in self._ee_results: continue
            aqua_dict = self._vqe_results['dicts'][pt]
            res_dict = self._vqe_results['results'][pt]
            deets = []
            deets.append(pt)
            deets.append(aqua_dict['algorithm']['name'])
            deets.append(aqua_dict['operator']['transformation'])
            deets.append(aqua_dict['operator']['qubit_mapping'])
            deets.append(aqua_dict['operator']['two_qubit_reduction'])
            deets.append(aqua_dict['variational_form']['name'])
            deets.append(aqua_dict['variational_form']['depth'])
            deets.append(aqua_dict['initial_state']['name'])
            deets.append(aqua_dict['optimizer']['name'])
            for line in res_dict['printable']:
                if re.search(re.compile(r'Total ground state energy'), line):
                    nums = re.findall(re.compile('[-\d.]+'), line)
                    deets.append(float(nums[-1]))
            deets.append(self._ee_results[pt]['energy'])
            deets.append(deets[-2]-deets[-1])
            for line in res_dict['printable']:
                if re.search(re.compile(r'Measured:: Num particles:'), line):
                    nums = re.findall(re.compile('[-\d.]+'), line)
                    deets.append(float(nums[-3]))
                    deets.append(float(nums[-2]))
                    deets.append(float(nums[-1]))
                if re.search(re.compile(r'(debye)'), line):
                    nums = re.findall(re.compile('[-\d.]+'), line)
                    deets.append(float(nums[-1]))
            deets.append(res_dict['algorithm_retvals']['eval_count'])
            deets.append(res_dict['algorithm_retvals']['eval_time'])
            results.append(deets)
        print(', '.join(["Qubit Mapping",	"2-Qubit Reduction",	"Variational Form",	"Depth",	"Initial State",
                         "Optimizer",	"VQE Ground State Energy (Hartree)",	"EE Ground State Energy (Hartree)",
                         "Energy Difference (Hartree)",	"Particle Number",	"Spin",	"M",	"Dipole Moment (debye)",
                         "Eval Count",	"Eval time (s)",]))
        for res in results:
            print(', '.join(map(str,res)))

    def guess_params(self,
                     point,
                     mode="last_point",
                     data_window_len = 3):

        # Create a dictionary of params by point for convenience
        self.sort_dicts()
        param_dict = {pt: np.array(res['opt_params']) for (pt, res) in self._vqe_results['results'].items() if
                        self._vqe_results['dicts'][pt]['variational_form'] == self._dicts[point]['variational_form'] and
                      pt < point}

        if mode == "no_guess" or len(param_dict) == 0: return None

        if data_window_len == -1 : data_window_len = len(param_dict)
        else: data_window_len = min(data_window_len, len(param_dict))
        window_pts = np.array(list(param_dict.keys()))[-data_window_len:]
        window_params = {pt: param_dict[pt] for pt in window_pts}

        # normal bootstrapping, take previous point's params as-is for current point.
        if mode == "last_point" or len(window_pts) == 1: return list(window_params[window_pts[-1]])

        return list(next_point_params)