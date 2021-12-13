#!/bin/bash

path_to_executable=$(which nccmp)
referenceFile=$1
goldFile=$2
saveFile=$3

if [[ ! -f $referenceFile ]]; then
    echo "Reference file $referenceFile does not exist"
    exit 1;
fi

if [[ -n $saveFile ]]; then
    echo "Gold file will be saved to $saveFile"
    cp $referenceFile $saveFile
fi

if [[ ! -f $goldFile ]]; then
    echo "Gold file $goldFile does not exist"
    exit 1;
fi

if [[ -x $path_to_executable ]];  then
    echo "Found nccmp: $path_to_executable"
else
    echo "ERROR: Cannot find nccmp in path"
    exit 1;
fi

# initialize the passflag 
passflag=0          # Initially set to "PASS"

STDTOL=5.0E-12 #1.0E-11      # Standard tolerance on the standard variables

# Variables to test
stdtestvars="velocity 
velocity_tavg
sfs_stress
resolved_stress
sfs_stress_tavg
temperature
temperature_tavg
temperature_sfs_flux_tavg
temperature_resolved_flux
temperature_resolved_flux_tavg
utau
"

# Test each of the standard variables one by one
for var in $stdtestvars; do
    echo "TESTING $var"
    nccmp -d -l -f -v $var -t $STDTOL $referenceFile $goldFile
    result=$?
    if [ "$result" -ne 0 ]; then
	echo "FAIL: $var"
	passflag=1
    fi
done

# Now do the temperature variance variables
# (And any other variables who need looser tolerances)
TEMPTOL=1.0E-9
tempvars="
temperature_variance
temperature_variance_tavg
resolved_stress_tavg
"

# Test each of the temperature variance variables one by one
for var in $tempvars; do
    echo "TESTING $var"
    nccmp -d -l -f -v $var -t $TEMPTOL $referenceFile $goldFile
    result=$?
    if [ "$result" -ne 0 ]; then
	echo "FAIL: $var"
	passflag=1
    fi
done

# Return the final result
exit $passflag
