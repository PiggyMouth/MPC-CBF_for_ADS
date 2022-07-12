[![DOI](https://zenodo.org/badge/505930246.svg)](https://zenodo.org/badge/latestdoi/505930246)
# MPC-CBF_for_ADS

The source codes for the work "Robust Safe Control for Automated Driving Systems With Perception Uncertainties" by Yan Feng Yu are listed here.

Simulation results are shown in the folder `sim_output`.

## Carla

The simulation is performed using [Carla simulator](https://github.com/carla-simulator/carla) version 0.9.12 with Python version 3.6.

## Scenarios

To run the simulation codes, please download [Scenario runner](https://github.com/carla-simulator/scenario_runner) provided by Carla.

The autonomous agents are based on KeyingLucyWang's repository [Safe_Reconfiguration_Scenarios](https://github.com/KeyingLucyWang/Safe_Reconfiguration_Scenarios). The least required files are `test_config.txt`, and files in the "navigation folder".

The scenario that is tested and validated is: `FollowLeadingVehicle_5`.

## Perception noise model

The perception noise model based on the computed distance by a CNN model during salt and pepper noise is stored in `differences.save`.
