import os
import math
import os.path


def exit_if_file_does_not_exist(fname):
    if not os.path.exists(fname):
        print 'No file: ' + fname
        exit(1)


def num_lines_output(fname):
    exit_if_file_does_not_exist(fname)
    f = open(fname)
    lines = f.readlines()
    header = lines[1].split()
    j = 0
    for line in lines[2:]:
        j += 1
        if line.split()[0] == header[0]:
            break
    f.close()
    return j


def tail(fname, n):
    exit_if_file_does_not_exist(fname)
    stdin, stdout = os.popen2("tail -n " + str(n) + " " + fname)
    stdin.close()
    lines = stdout.readlines()
    stdout.close()
    return lines


def extract_norm(lines):
    l2norm = {}
    numNodes = lines[0].split()[3]

    for line in lines:
        tokens = line.split()
        l2norm[tokens[0]] = float(tokens[6])

    return numNodes, l2norm


def compute_and_check_ooas(basep_name, numResolutions, dim, min_ooas):
    first_name = basep_name + "_R0.dat"
    linesToRead = num_lines_output(first_name)
    nodes0, norm0 = extract_norm(tail(first_name, linesToRead))

    for j in xrange(numResolutions - 1):
        name = basep_name + "_R" + str(j + 1) + ".dat"
        nodes1, norm1 = extract_norm(tail(name, linesToRead))
        order = {}
        for varname, norm in norm1.iteritems():
            num_ratio = math.log(float(norm1[varname]) / float(norm0[varname]))
            order[varname] = num_ratio / math.log(2.0)

        norm0 = norm1
        nodes0 = nodes1

        output_string = "R" + str(j + 1) + "-" + str(j)

        for varname, ooa in order.iteritems():
            output_string += " " + varname + ": " + str(ooa)
            if ooa > min_ooas[varname]:
                print 'Bad convergence'
                print output_string
                exit(1)



dimension = 3
numResolution = 2
base_name = 'BoussinesqNonIso'

min_allowed_ooas = {'velocity[0]': -1.95, 'velocity[1]': -1.95, 'velocity[2]': -1.95, 'temperature[0]': -1.95}

compute_and_check_ooas(base_name, numResolution, dimension, min_allowed_ooas)

exit(0)
