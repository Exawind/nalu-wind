#!/usr/bin/env python3

"""
Check L2 norms
"""

import os
import os.path
import argparse
from shutil import copyfile


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Nalu-Wind verification test check utility")
    parser.add_argument(
        "test_name", help="Verification test name")
    parser.add_argument(
        "gold_norms", help="Absolute path to the gold norms file")
    parser.add_argument(
        '--abs-tol', type=float, default=1.0e-10,
        help="Tolerance for absolute error")
    parser.add_argument(
         '--save-norm-file', required=False,
         help="File in which to save a copy of the norms")
    return parser.parse_args()


def exit_if_file_does_not_exist(fname):
    if not os.path.exists(fname):
        print ('No file: ' + fname)
        exit(1)


def compute_and_check_norms(base_name, gold_norm_name, tol):
    norm_name = base_name + ".dat"
    exit_if_file_does_not_exist(norm_name)
    args = parse_arguments()
    if (args.save_norm_file != None):
        copyfile(norm_name, args.save_norm_file)
    f_norm = open(norm_name,'r')
    norm_lines = f_norm.readlines()[2:]
    
    f_gold_norm = open(gold_norm_name,'r')
    gold_norm_lines = f_gold_norm.readlines()[2:]
    
    if len(norm_lines) != len(gold_norm_lines):
        print ('Number of timesteps do not match')
        exit(1)
        
    for i in range(1, len(norm_lines)):
        norm = norm_lines[i].split()
        gold_norm = gold_norm_lines[i].split()
        
        for j in range(3, len(norm)):
            abs_diff = abs(float(norm[j]) - float(gold_norm[j]))
            if abs_diff > tol:
                print ('Solution norm does not match for ' + gold_norm[0] + 
                       ' at time step ' + gold_norm[1] + 
                       ' with absolute error ' + str(abs_diff))
                exit(1)


args = parse_arguments()
compute_and_check_norms(args.test_name, args.gold_norms, args.abs_tol)
exit(0)
