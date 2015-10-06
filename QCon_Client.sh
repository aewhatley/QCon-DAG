#!/bin/bash

#
# Run the MapReduce module as a standalone client
#

#set -x

LST=$1
PVE=$2
shift 2

if [[ `hostname -f` = *brazos.tamu.edu ]]; then
    export LD_LIBRARY_PATH=$1
    shift 1
    MR=~/PythonModules/Utilities/MapReduce.py
else
    MR=~/Dropbox/PythonModules/Utilities/MapReduce.py
fi

NCLIENTS=$1
MOM=$2
shift 2

if [[ ${OSTYPE} = linux* ]]; then
    PRE="/usr/bin/time --verbose"
else if [[ ${OSTYPE} = darwin* ]]; then
    PRE="/usr/bin/time -l"
fi; fi

. $PVE

ME=`uname -n`

if [[ "$ME" = "alquerque" || "$ME" = "sudoku" ]]; then
    export QCON_Map_Mem_Threshold=750000000
    export QCON_Reduce_Interm_Write=4000000
else if [[ `hostname -f` = *brazos.tamu.edu ]]; then
    export QCON_Map_Mem_Threshold=1500000000
    export QCON_Reduce_Interm_Write=30000000
fi; fi

function RunClient() {

    echo "************************ Starting $NCLIENTS clients on node $ME at `date +'%D %T.%N'`"

    $PRE $MR -p QCon -n $NCLIENTS $MOM $@

    echo "************************ Finished $NCLIENTS clients on node $ME at `date +'%D %T.%N'`"

}

RunClient $@ &> $LST
