#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Check mean system norm errors in regression tests

This script determines the pass/fail status of a regression test by comparing
the "Mean System Norm" values output at each timestep against "gold values"
from the reference file provided by the user.

Success is determined by the following criteria: the number of timesteps in the
log file matches the number of timesteps in the gold file, and for each
timestep the system norms meet the absolute and relative tolerances (default
1.0e-16 and 1.0e-7 respectively). The tolerances can be adjusted using command
line arguments, pass `-h` to get a brief usage message.
"""

import sys
import os
import math
import subprocess
import argparse
from shutil import copyfile

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Nalu-Wind regression test check utility")
    parser.add_argument(
        '--abs-tol', type=float, default=1.0e-15,
        help="Tolerance for absolute error")
    parser.add_argument(
        '--rel-tol', type=float, default=1.0e-7,
        help="Tolerance for relative error")
    parser.add_argument(
        "test_name", help="Regression test name")
    parser.add_argument(
        "gold_norms", help="Absolute path to the gold norms file")
    parser.add_argument(
        '--save-norm-file', required=False,
        help="File in which to save a copy of the norms")
    return parser.parse_args()

def load_norm_file(fname):
    """Parse the norm file and return the mean system norms"""
    try:
        with open(fname, 'r') as fh:
            lines = fh.readlines()
            norms = [float(ll.strip().split()[0]) for ll in lines]
            return norms
    except:
        return []

def generate_test_norms(testname):
    """Parse the log file and generate test norms"""
    logname = testname + ".log"
    norm_name = testname + ".norm"
    cmdline = """awk '/Mean System Norm:/ { print $4, $5, $6; }' %s > %s """%(
        logname, norm_name)
    os.system(cmdline)
    args = parse_arguments()
    if (args.save_norm_file != None):
        copyfile(norm_name, args.save_norm_file)

    return load_norm_file(norm_name)

def get_run_time(testname):
    """Return STKPERF total time"""
    logname = testname + ".log"
    cmdline = """awk '/STKPERF: Total Time/ { print $4; }' %s """%(
        logname)
    try:
        pp = subprocess.run(cmdline, shell=True, check=True, capture_output=True)
        return pp.stdout.decode('UTF-8').strip()
    except:
        return ""

def check_norms(test_norms, gold_norms, atol, rtol):
    """Check the regression test norms"""
    if len(test_norms) != len(gold_norms):
        print("Number of timesteps do not match", flush=True)
        return (False, 1.0e16, 1.0e16)

    test_pass = True
    abs_diff = 0.0
    rel_diff = 0.0

    for t1, t2 in zip(test_norms, gold_norms):
        adiff = abs(t1 - t2)
        rdiff = abs(t1 / t2 - 1.0)

        abs_diff = max(abs_diff, adiff)
        rel_diff = max(rel_diff, rdiff)

        if (adiff > atol) and (rdiff > rtol):
            test_pass = False

    return (test_pass, abs_diff, rel_diff)

def main():
    """Driver function"""
    args = parse_arguments()
    test_norms = generate_test_norms(args.test_name)
    gold_norms = load_norm_file(args.gold_norms)
    run_time = get_run_time(args.test_name)
    run_time = float(run_time) if run_time else 0.0
    status, adiff, rdiff = check_norms(
        test_norms, gold_norms, args.abs_tol, args.rel_tol)

    name = args.test_name.ljust(40, ".")
    status_str = "PASS:" if status else "FAIL:"
    print("%s %-40s %10.4fs %.4e %.4e"%(
        status_str, name, run_time, adiff, rdiff), flush=True)
    sys.exit(0 if status else 1)

if __name__ == "__main__":
    main()
