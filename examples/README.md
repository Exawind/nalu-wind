# Nalu example files

This directory contains different nalu-wind example cases

1. abl_neutral
2. turbine_alm

Each case has an example 'input_files' directory.
 The script `nalu_input_fileX` will take as an input a setup file
(`default=set_up.yaml`) and it will modify the input_files to reflect all the settings
specified in the `set_up.yaml` file.
The new files are written and can be used by the Nalu sover.

## Step 1

Create a conda environment to be able to use the python script.
This step needs to be done only once to create a python environment with the neccessary packages.
```
conda create -n nalu_python -c conda-forge python=3.6 numpy ruamel.yaml
```

To activate this environment, use:
```
source activate nalu_python
```

To deactivate an active environment, use:
```
source deactivate
```

## Step 2

Go into the case directory and run the script nalu_input_fileX to generate input files.
Assuming you are in the directiory nalu-wind/examples:

```cd abl_neutral
../nalu_input_fileX -s set_up.yaml
```

## Step 3

Run the simulation:
```
mpirun -np 600 naluX -i abl_simulation.yaml
```
