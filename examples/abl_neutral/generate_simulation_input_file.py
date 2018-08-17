#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  generate_simulation_file_input.py
#  
#  Copyright 2018 Martinez <lmartin1@LMARTIN1-31527S>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
from ruamel.yaml import YAML
import numpy as np

def main(args):
    '''
    This script will load a yaml input file will add the necessary inputs for
    an atmospheric boundary layer simulation using Nalu.
    
    File input for users to modify:
        setUp.yaml
        
    File input for advanced users:
        file_inputs/ablNeutralEdge.yaml
        
    File output (do not modify):
        ablSimulation.yaml

    '''
    # Define all the yaml input properties
    yaml=YAML(typ='rt')   # 'rt' (round-trip)
    # Specify file indentation
    yaml.indent(mapping=2, sequence=2, offset=0)
    # mapping is the indentation for dictionaries
    yaml.default_flow_style = False

    '''
    Inputs
    '''
    # This file is the one that specifies all the inputs for a wind turbine
    #     simulation. 
    # This is similar to the SOWFA setUp file used to specify important 
    #     simulation details
    file_setup = 'setUp.yaml'
    # Load all the yaml inputs into from the setup file
    yaml_setup = yaml.load(open(file_setup))

    # The original input file which will be used as a guideline
    # Note that this file should not be modified to run simulations. 
    # Advanced user options are contained in this file.
    file_input  = 'input_files/ablNeutralEdge.yaml'
    # Load all the yaml inputs into this variable
    yaml_input = yaml.load(open(file_input))

    # The preprocessing file used to create the initial condition
    file_preprocess = 'input_files/nalu_preprocess.yaml'
    # Load all the yaml inputs into this variable
    yaml_preprocess = yaml.load(open(file_preprocess))

    '''
    Outputs
    '''
    # This is the name of the output file that Nalu will read at simulation time
    file_output = 'abl_simulation.yaml'
    file_output_preprocess = 'abl_preprocess.yaml'



    '''
    Modify the pre_processing file
    Most modification are here, but some modifications are done further down
    '''
    # Add the mesh the preproceccing only if it exists
    if yaml_setup['mesh']['generate'] == 'yes':
        # Get the mesh file name        
        yaml_preprocess['nalu_abl_mesh']['output_db'] = \
            yaml_setup['mesh']['mesh_file']
        yaml_preprocess['nalu_abl_mesh']['vertices'] = \
            yaml_setup['mesh']['vertices']
        yaml_preprocess['nalu_abl_mesh']['mesh_dimensions'] = \
            yaml_setup['mesh']['mesh_dimensions']
    else:
        # Delete the mesh entry from the pre-processing file
        del(yaml_preprocess['nalu_abl_mesh'])

    # Set the preprocess mesh
    yaml_preprocess['nalu_preprocess']['input_db'] = \
        yaml_setup['mesh']['mesh_file']
    yaml_preprocess['nalu_preprocess']['output_db'] = \
        yaml_setup['mesh']['mesh_file']

    # Setup the initial ondition in the pre-process
    #~ yaml_preprocess['nalu_preprocess']['init_abl_fields'] = \
    #~ yaml_setup['mesh']['mesh_file']
    

    ############################################################################
    #     This will pick up all the needed variables from the setUp.i file     #
    #         and modify the Nalu input files accordingly                      #
    ############################################################################
    # The velocity field at hub height
    U0Mag = yaml_setup['U0Mag']
    wind_height = yaml_setup['wind_height']
    wind_dir = np.deg2rad(yaml_setup['wind_dir'])

    # The wind height as a list
    yaml_input['realms'][0]['abl_forcing']['momentum']['heights'] = \
        [wind_height]

    # The velocity components
    # The time history of the velocity components
    # Start at zero and have a very large number for the last time
    t0 = 0.    # Time to start forcing
    t1 = 1.e9  # Time to end forcing (large number)
    wind_x = U0Mag * np.cos(wind_dir)
    yaml_input['realms'][0]['abl_forcing']['momentum']['velocity_x'] = [
            [t0, wind_x],
            [t1, wind_x]]
    wind_y = U0Mag * np.sin(wind_dir)
    yaml_input['realms'][0]['abl_forcing']['momentum']['velocity_y'] = [
            [t0, wind_y],
            [t1, wind_y]]
    wind_z = 0.
    yaml_input['realms'][0]['abl_forcing']['momentum']['velocity_z'] = [
            [t0, wind_z],
            [t1, wind_z]]


    #######################################
    #          Initial Conditions         #
    #######################################
    yaml_preprocess['nalu_preprocess']['init_abl_fields']['velocity'] \
        ['heights'] = [wind_height]
    yaml_preprocess['nalu_preprocess']['init_abl_fields']['velocity'] \
        ['values'] = [wind_x, wind_y, wind_z]

    yaml_preprocess['nalu_preprocess']['init_abl_fields']['temperature'] \
        ['heights'] = yaml_setup['temperature_heights']
    yaml_preprocess['nalu_preprocess']['init_abl_fields']['temperature'] \
        ['values'] = yaml_setup['temperature_values']

    #######################################
    #         Boundary Conditions         #
    #######################################
    # Loop through all the boundary conditions and modify accordingly
    for var in yaml_input['realms'][0]['boundary_conditions']:

        # Top boundary condition
        if 'symmetry_boundary_condition' in var.keys():
            var['symmetry_user_data']['normal_temperature_gradient'] = \
                yaml_setup['TGradUpper']
            print('Modified top boundary condition')

        # Bottom wall boundary condition
        if 'wall_boundary_condition' in var.keys():
            var['wall_user_data']['roughness_height'] = \
                yaml_setup['z0']
            var['wall_user_data']['reference_temperature'] = \
                yaml_setup['TRef']
            var['wall_user_data']['heat_flux'] = \
                yaml_setup['qwall']
            print('Modified lower boundary condition')



    #######################################
    #         Material Properties         #
    #######################################
    # The density
    for var in yaml_input['realms'][0]['material_properties']['specifications']:
        if var['name'] == 'density':
            var['value'] = yaml_setup['density']
    for var in yaml_input['realms'][0]['solution_options']['options']:
        if 'user_constants' in var:
            var['user_constants']['reference_density'] = yaml_setup['density']

    # Kinematic viscosity
    for var in yaml_input['realms'][0]['material_properties']['specifications']:
        if var['name'] == 'viscosity':
            var['value'] = yaml_setup['nu']

    #########################
    #   ABL Properties      #
    #########################
    # Loop through all the options and find user_constants
    for var in yaml_input['realms'][0]['solution_options']['options']:
        if 'user_constants' in var:

            # The latitude
            var['user_constants']['latitude'] = yaml_setup['latitude']
            # The reference temperature
            var['user_constants']['reference_temperature'] = yaml_setup['TRef']

    #########################
    #          Time         #
    #########################
    yaml_input['Time_Integrators'][0]['StandardTimeIntegrator'] \
        ['termination_step_count'] = yaml_setup['termination_step_count']
    yaml_input['Time_Integrators'][0]['StandardTimeIntegrator'] \
        ['time_step'] = yaml_setup['time_step']


    '''
    Save the new Nalu input files
    '''
    # The preprocessing file
    yaml.dump(yaml_preprocess, open(file_output_preprocess, 'w'))
    # The input file for running naluX
    yaml.dump(yaml_input, open(file_output,'w'), )

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
