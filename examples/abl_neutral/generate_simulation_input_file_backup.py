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
import argparse

 
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
    
    # Initialize the mapping object
    mapping_object = mapping_object_class(setup_file='setUp.yaml', )



class mapping_object_class():
    '''
    A class used to mapp the yaml setup file to Nalu inputs
    '''
    def __init__(self, 
        ##############################
        # Input files which are read #
        ##############################
        # The setup file
        setup_file='setUp.yaml',
        # The input file used as a starting point
        example_input_file='input_files/ablNeutralEdge.yaml',
        # The pre-processing input file
        example_preprocess_file='input_files/nalu_preprocess.yaml',
        #################################
        # Input files which are written #
        #################################
        # The output file that has been modified from example_input_file
        nalu_input_file_name='abl_simulation.yaml',
        # The pre-processing file which has been modified from 
        #     example_preprocess_file
        preprocess_file_name='abl_preprocess.yaml'
        ):

        '''
        Initialize the object
        '''

        # Open the input file as a yaml file
        self.yaml_setup = self.open_yaml_file(setup_file)

        # The dictionary containing the entries of the setup file and the
        #   functions to map the setup inputs to the nalu input file
        self.mapping_dictionary = {
            # Wind speed and temperature profiles
            ('U0Mag', 'wind_dir', 'wind_height'): self.set_velocity,
            'temperature_heights': self.set_temperature_heights,
            'temperature_values': self.set_temperature_values,
            # Material Properties
            'density': '',
            'nu': '',
            'TRef': '',
            'latitude': '',
            # Bottom Wall
            'qwall': '',
            'z0': '',
            # Time controls
            'time_step': '',
            'total_run_time': '',
            # Mesh
            'mesh': '',
            # Output
            'output_frequency': '',
            'output_data_base_name': '',
            # Boundary layer statistics
            'boundary_layer_statistics': '',
        }

    def write_output_files(self):
        '''
        Write the output files
        '''
        # Define all the yaml input properties
        yaml=YAML(typ='rt')   # 'rt' (round-trip)
        # Specify file indentation
        yaml.indent(mapping=2, sequence=2, offset=0)
        # mapping is the indentation for dictionaries
        yaml.default_flow_style = False

        # The preprocessing file
        yaml.dump(self.preprocess_file_name, open(file_output_preprocess, 'w'))
        # The input file for running naluX
        yaml.dump(self.nalu_input_file_name, open(file_output,'w'), )

    def set_velocity(self):
        '''
        Function to setup the velocity entries in the file

        This operation takes the wind angle and converts it to the proper 
             cartesian coordinate system.
        A direction of 270 deg means the wind is coming from the west, which
             is from left to right.
        
                              N 0deg
                               |
                               |
                               |
                     W 270deg --- E 90deg
                               |
                               |
                               |
                              S 180deg
        '''

        # The velocity field at hub height
        U0Mag = self.yaml_setup['U0Mag']
        wind_height = self.yaml_setup['wind_height']
        wind_dir = np.deg2rad(270. - self.yaml_setup['wind_dir'])

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

    def set_temperature_heights(self):
        '''
        Set the temperature heights
        '''

    def set_temperature_values(self):
        '''
        Set the temperature heights
        '''


    @staticmethod
    def open_yaml_file(file_input):
        '''
        Open the yaml file
        file_input = the input file
        Returns the opened yaml file
        yaml_file = the open yaml file
        '''

        # Define all the yaml input properties
        yaml=YAML(typ='rt')   # 'rt' (round-trip)
        # Specify file indentation
        yaml.indent(mapping=2, sequence=2, offset=0)
        # mapping is the indentation for dictionaries
        yaml.default_flow_style = False

        # The output file
        yaml_file = yaml.load(open(file_input))

        return yaml_file





def original_main(args):
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

    # The original input file which will be used as a guideline.
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
            
        # The domain size
        # Lower bottom coordinate
        yaml_preprocess['nalu_abl_mesh']['vertices'][0][0] = \
            yaml_setup['mesh']['domain_bounds_x'][0]
        yaml_preprocess['nalu_abl_mesh']['vertices'][0][1] = \
            yaml_setup['mesh']['domain_bounds_y'][0]
        yaml_preprocess['nalu_abl_mesh']['vertices'][0][2] = \
            yaml_setup['mesh']['domain_bounds_z'][0]
        # Upper coordinate
        yaml_preprocess['nalu_abl_mesh']['vertices'][1][0] = \
            yaml_setup['mesh']['domain_bounds_x'][1]
        yaml_preprocess['nalu_abl_mesh']['vertices'][1][1] = \
            yaml_setup['mesh']['domain_bounds_y'][1]
        yaml_preprocess['nalu_abl_mesh']['vertices'][1][2] = \
            yaml_setup['mesh']['domain_bounds_z'][1]

        # Number of grid points
        yaml_preprocess['nalu_abl_mesh']['mesh_dimensions'] = \
            yaml_setup['mesh']['number_of_cells']

    else:

        # Delete the mesh entry from the pre-processing file
        del(yaml_preprocess['number_of_grid_points'])

    # Set the preprocess mesh
    yaml_preprocess['nalu_preprocess']['input_db'] = \
        yaml_setup['mesh']['mesh_file']
    yaml_preprocess['nalu_preprocess']['output_db'] = \
        yaml_setup['mesh']['mesh_file']

    # Setup the initial ondition in the pre-process
    yaml_preprocess['nalu_preprocess']['input_db'] = \
    yaml_setup['mesh']['mesh_file']
    yaml_preprocess['nalu_preprocess']['output_db'] = \
    yaml_setup['mesh']['mesh_file']

    # Set the mesh file in the simulation file
    yaml_input['realms'][0]['mesh'] = yaml_setup['mesh']['mesh_file']

    ############################################################################
    #     This will pick up all the needed variables from the setUp.i file     #
    #         and modify the Nalu input files accordingly                      #
    ############################################################################
    # The velocity field at hub height
    U0Mag = yaml_setup['U0Mag']
    wind_height = yaml_setup['wind_height']
    # This operation takes the wind angle and converts it to the proper 
    #     cartesian coordinate system.
    # A direction of 270 deg means the wind is coming from the west, which is
    #     from left to right.
    #
    #
    #                      N 0deg
    #                       |
    #                       |
    #                       |
    #             W 270deg --- E 90deg
    #                       |
    #                       |
    #                       |
    #                      S 180deg
    #
    #
    wind_dir = np.deg2rad(270. - yaml_setup['wind_dir'])

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
        ['heights'] = [0, wind_height]
    yaml_preprocess['nalu_preprocess']['init_abl_fields']['velocity'] \
        ['values'] = [[wind_x, wind_y, wind_z],[wind_x, wind_y, wind_z],]

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

            # If the gradient is specified, use that value, if not, compute it
            #     from the initial condition
            try:
                # Read the upper wall temperature gradient
                var['symmetry_user_data']['normal_temperature_gradient'] = \
                    yaml_setup['TGradUpper']

            except:

                # Compute the gradient based on the initial condition
                print('TGradUpper computed from initial condition')
                TGradUpper = (yaml_setup['temperature_values'][-1] 
                                - yaml_setup['temperature_values'][-2]) / (
                                yaml_setup['temperature_heights'][-1] 
                                - yaml_setup['temperature_heights'][-2])
                # Assign the initial condition
                var['symmetry_user_data']['normal_temperature_gradient'] = \
                    TGradUpper

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
    ################################### Change
    for var in yaml_input['realms'][0]['material_properties']['specifications']:
        if var['name'] == 'viscosity':
            var['value'] = yaml_setup['nu']

    # Set the reference pressure
    R = yaml_input['realms'][0]['material_properties'] \
        ['constant_specification']['universal_gas_constant']
    yaml_input['realms'][0]['material_properties']['constant_specification']   \
        ['reference_pressure'] = R * yaml_setup['density'] * yaml_setup['TRef']

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

    # Boundary layer statistics
    yaml_input['realms'][0]['boundary_layer_statistics']['stats_output_file'] \
        = yaml_setup['boundary_layer_statistics']['stats_output_file']
    #~ yaml_input['realms'][0]['boundary_layer_statistics']['output_frequency'] \
        #~ = yaml_setup['boundary_layer_statistics']['output_frequency']
    yaml_input['realms'][0]['boundary_layer_statistics']                       \
        ['time_hist_output_frequency'] \
        = yaml_setup['boundary_layer_statistics']['time_hist_output_frequency']

    #########################
    #          Time         #
    #########################
    yaml_input['Time_Integrators'][0]['StandardTimeIntegrator'] \
        ['time_step'] = yaml_setup['time_step']
    # Total number of time-steps to run
    yaml_input['Time_Integrators'][0]['StandardTimeIntegrator'] \
        ['termination_step_count'] = int(yaml_setup['total_run_time']
            / yaml_setup['time_step'])

    #################################
    #          Output files         #
    #################################
    yaml_input['realms'][0]['output']['output_data_base_name'] = \
        yaml_setup['output_data_base_name']
    yaml_input['realms'][0]['output']['output_frequency'] = \
        yaml_setup['output_frequency']


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
