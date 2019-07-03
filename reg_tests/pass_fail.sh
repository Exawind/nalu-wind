#!/bin/bash

# Compute absolute and relative diffs of gold norms using bc
#
# Returns a string that can be evaluated to be a bash array
# ( abs_diff, rel_diff)
#
compute_diffs() {
    local currNorm=$1
    local goldNorm=$2

    bc -l <<EOF
    c = $currNorm
    g = $goldNorm
    a = (c - g)
    if (a < 0.0) a = -a
    r = (c / g - 1.0)
    if (r < 0.0) r = -r
    print "( ", a, " ", r, " )"
EOF
}

determine_pass_fail() {

    diffAnywhere=0
    tolerance=$1
    logFileName=$2
    localNormFileName=$3
    goldNormFileName=$4

    # check for required files: log file and gold
    if [ ! -f "$logFileName" ]; then
        diffAnywhere=1
    fi

    # check for gold norm file
    if [ ! -f "$goldNormFileName" ]; then
        diffAnywhere=1
    fi

    grep "Mean System Norm:" "$logFileName"  | awk '{print $4 " " $5 " " $6 }' > "$localNormFileName"

    # make sure the grep  worked
    if [ ! -f "$localNormFileName" ]; then
        diffAnywhere=1
    fi

    # read in gold norm values
    goldCount=1
    goldFileContent=( `cat "$goldNormFileName"`)
    for gfc in "${goldFileContent[@]}" ; do
        goldNorm[goldCount]=$gfc
        ((goldCount++))
    done

    # read in local norm values
    localCount=1
    localFileContent=( `cat "$localNormFileName"`)
    for lfc in "${localFileContent[@]}" ; do
        localNorm[localCount]=$lfc
        ((localCount++))
    done

    if [ $(echo " $goldCount - $localCount" | bc) -eq 0 ]; then
        # the lengths the same... proceed
        for ((i=1;i<$goldCount;i+=3)); do
            modLocalNorm=$(printf "%1.32f" ${localNorm[i]})
            modGoldNorm=$(printf "%1.32f" ${goldNorm[i]})

            eval "diffs=$(compute_diffs $modLocalNorm $modGoldNorm)"
            absDiff=${diffs[0]}
            relDiff=${diffs[1]}

            # test the difference
            if [ $(echo " $absDiff > $tolerance" | bc) -eq 1 ]; then
                diffAnywhere=1
            fi

            # find the max
            if [ $(echo " $absDiff > $maxSolutionDiff " | bc) -eq 1 ]; then
                maxSolutionDiff=$absDiff
                maxRelDiff=$relDiff
            fi

        done

    else
        # length was not the same; fail
        diffAnywhere=1
        maxSolutionDiff=1000000.0
    fi

    # extract simulation time
    return $diffAnywhere  

}

# $1 is the test name
main() {
  testName=$1
  goldFile=$2
  mytolerance=$3
  maxSolutionDiff=-1000000000.0
  maxRelDiff=-1000000000.0
  determine_pass_fail ${mytolerance} "${testName}.log" "${testName}.norm" "${goldFile}"
  passStatus="$?"
  performanceTime=`grep "STKPERF: Total Time" ${testName}.log  | awk '{print $4}'`
  padding="........................................"
  if [ ${passStatus} -eq 0 ]; then
      printf "PASS: %s%s %10.4fs\n" ${testName} ${padding:${#testName}} ${performanceTime}
      exit 0
  else
      printf "FAIL: %s%s %10.4fs %.4e %.4e\n" ${testName} ${padding:${#testName}} ${performanceTime} ${maxSolutionDiff} ${maxRelDiff}
      exit 1
  fi
}

# ./pass_fail.sh testName normFile tolerance
main "$@"
