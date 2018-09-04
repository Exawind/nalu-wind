# This is a pre-processing file to create a mesh and initial condition
# The file requires installation of nalu-wind and nalu-wind-utils

# Run the script to generate the input files
# This scripts reads the inputs in setUp.yaml and creates the correct
#    input files for Nalu based on the test case provided in the directory
#    'input_files'
#python generate_simulation_input_file.py

# Create the mesh
# This will read the mesh inputs form the file and generate the mesh
#~/Software/NaluWindUtils/build/src/mesh/abl_mesh -i abl_preprocess.yaml

# Initialize the inflow
~/Software/NaluWindUtils/build/src/preprocessing/nalu_preprocess -i abl_preprocess.yaml
