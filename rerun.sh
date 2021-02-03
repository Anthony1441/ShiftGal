#!/bin/bash

while read p; do
    if [[ `echo "$p < 650" | bc -l` -gt 0 ]]; then
        ( ./dirRun.sh $p $p >/dev/null 2>&1 & )
    fi
done < ../rerun.txt
