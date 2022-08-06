[![DOI](https://zenodo.org/badge/505930246.svg)](https://zenodo.org/badge/latestdoi/505930246)
# MPC-CBF_for_ADS

The source codes for the work "Robust Safe Control for Automated Driving Systems With Perception Uncertainties" by Yan Feng Yu are listed here.

Simulation results are shown in the folder `sim_output` (at the moment when Kalman filter is placed after the control action). 

The scripts: 
* `acc_with_cbf_mpc.py` simulate the case without any perception noise present
* `acc_with_cbf_mpc_noise.py` simulates the case with perception noise present
* `acc_with_cbf_mpc_kf.py` simulates with Kalman filter placed before the control action to handle the perception noise


## Carla

The simulation is performed using [Carla simulator](https://github.com/carla-simulator/carla) version 0.9.12 with Python version 3.6.

## Scenarios

The scenario is based on [Scenario runner](https://github.com/carla-simulator/scenario_runner), an module provided by CARLA.

The autonomous agents are based on KeyingLucyWang's repository [Safe_Reconfiguration_Scenarios](https://github.com/KeyingLucyWang/Safe_Reconfiguration_Scenarios).

The simulation provided here is tested and validated on scenario: `FollowLeadingVehicle_5`.

## Perception noise model

The perception noise model based on the computed distance by a CNN model during salt and pepper noise is stored in `differences.save`.

## Getting started

To run the simulation, begin with launching CARLA (CarlaUE4.sh or CarlaUE4.exe).

Next, open two terminals where you export the following path that suits your own computer (according to [Scenario runner: Getting started](https://carla-scenariorunner.readthedocs.io/en/latest/getting_scenariorunner/)):
```
export CARLA_ROOT=/path/to/your/carla/installation
export SCENARIO_RUNNER_ROOT=/path/to/your/scenario/runner/installation
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-<VERSION>.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
```
In my computer where Windows version of CARLA is used, it becomes:
```
set CARLA_ROOT=C:\Users\Admin\Simplepath\IL2232\CARLA_0912\WindowsNoEditor
set SCENARIO_RUNNER_ROOT=C:\Users\Admin\Simplepath\Exjobb\MPC-CBF_for_ADS
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla\dist\carla-0.9.12-py37-win-amd64.egg
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla\agents
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla\agents\navigation
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla\agents\tools
```
In the first terminal, enter the command: `python scenario_runner.py --scenario FollowLeadingVehicle_5 --reloadWorld`.  
In the second terminal, enter the python script name that you want to run i.e.: `python acc_with_cbf_mpc_kf.py`.

You will now see a pygame window with the scenario running. To run it again, simply enter the aforementioned commands again. 

The figures in the folder `sim_output` stores the case when N = 6, 8, 12. For respective case, T is preferred to be T = 12, 8, 6.
