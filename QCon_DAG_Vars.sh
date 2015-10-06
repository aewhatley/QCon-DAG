#
# Variables used by the QCon jobs
#

# ------------------------------------------------ List of clients

if [[ `hostname -f` = *brazos.tamu.edu ]]; then
    CLIENTFILE=$PBS_NODEFILE
else if [[ ${OSTYPE} = linux* ]]; then
    CLIENTFILE=`mktemp -t QConCFILEXXX`
    cat <<EOF > $CLIENTFILE
alquerque
alquerque
alquerque
alquerque
sudoku
sudoku
sudoku
sudoku
EOF
else if [[ ${OSTYPE} = darwin* ]]; then
    CLIENTFILE=`mktemp -t QConCFILEXXX`
fi; fi; fi

# ------------------------------------------------ Constants

PVE=$WORKON_HOME/p3.3/bin/activate

WDBNAME=Work
RDBNAME=Results

CNODE=`hostname`
IFS=. read ME STUFF <<< "$CNODE"

if [[ ${OSTYPE} = linux* ]]; then
    PRE="/usr/bin/time --verbose"
else if [[ ${OSTYPE} = darwin* ]]; then
    PRE="/usr/bin/time -l"
fi; fi

# ------------------------------------------------ SetVars()

function SetVars() {

    if [[ `hostname -f` = *brazos.tamu.edu ]]; then
	JOBNAME=$PBS_JOBNAME
	IFS=. read JOBID STUFF <<< "$PBS_JOBID"
	QC=~/QCon_DAG/QCon_DAG.py
	OUTDIR=/data/aewhatley/output
	MONGO=~/packages/mongodb-linux-x86_64-2.4.8
	MDBDIR=/data/aewhatley/mongo
    else
	BN=$(basename "$TREEFILE")
	FN=${BN%.*}
	NT=`printf "%05d" $NTREES`
	JOBNAME=${FN//-/_}_TR$NT
	JOBID=$$
	QC=~/Dropbox/QCon/QCon.py
	if [[ ${OSTYPE} = linux* ]]; then
	    OUTDIR=~/Shared/Quartet/Results/$OUTSET/Output
	    MONGO=~/Shared/Packages/mongodb/mongodb-linux-x86_64-2.4.7
	    MDBDIR=~/mongo
	else
	    OUTDIR=~/Quartet/Results/$OUTSET/Output
	    MONGO=~/Development/mongodb/mongodb-osx-x86_64-2.4.7
	    MDBDIR=~/Development/mongodb
	fi
    fi

    MDBDIR=${MDBDIR}/$ME
    mkdir -p ${MDBDIR}/data

}

# ------------------------------------------------ StartDB()

function StartDB() {

    CONFIG=${MDBDIR}/mongo.cfg

    cat <<EOF > $CONFIG
logpath = ${MDBDIR}/mongo.log
logappend = true
fork = true
dbpath = ${MDBDIR}/data
setParameter = logLevel=0
profile=0
slowms=300
EOF

    ${MONGO}/bin/mongod -f $CONFIG
    if [[ $? -ne 0 ]]; then 
	echo "Unable to start database, exiting" 
	exit 1
    fi

}

# ------------------------------------------------ StopDB()

function StopDB() {

    pkill mongod

    MPID=`pgrep mongod`
    while [ "$MPID" != "" ]; do
	sleep 1
	MPID=`pgrep mongod`
    done

    echo "MongoDB terminated"

}

# ------------------------------------------------ DeleteDB()

function DeleteWorkDB() {

    ${MONGO}/bin/mongo $WDBNAME <<EOF
db.dropDatabase()
EOF

}

# ------------------------------------------------ UnloadDB()

function UnloadDB() {

    DBPATH="--dbpath ${MDBDIR}/data"
    DBPATH=

    ${MONGO}/bin/mongoexport $DBPATH -d $RDBNAME -c ${JOBNAME}_Results | bzip2 > ${OUTDIR}/${JOBNAME}_Results.bz2
    ${MONGO}/bin/mongoexport $DBPATH -d $RDBNAME -c ${JOBNAME}_Stats   | bzip2 > ${OUTDIR}/${JOBNAME}_Stats.bz2

}

# ------------------------------------------------ StartClients()

function StartClients() {


    if [[ `hostname -f` = *brazos.tamu.edu ]]; then
	CMETHOD="/usr/local/bin/pbsdsh -h"
	CLIENT=~/QCon_DAG/QCon_Client.sh
	EXTRA=$LD_LIBRARY_PATH
    else
	CMETHOD=ssh
	CLIENT=~/Dropbox/QCon_DAG/QCon_Client.sh
    fi

    CLIST=`mktemp -t QConCListXXX`

    awk -v me="$ME" '

BEGIN {
    node[""] = 0;
}

{
    if ($1 != me) {
        node[$1]++;
    }
}

END {
    for (i in node) {
        if ( i != "") {
            printf("%s,%d\n", i, node[i])
        }
    }
}

' $CLIENTFILE > $CLIST

    LINES=$(cat $CLIST)
    for LINE in $LINES; do

	NODE=${LINE%,*}
	NCLIENTS=${LINE#*,}

	echo "Starting $NCLIENTS clients on node $NODE"

	$CMETHOD $NODE $CLIENT ${OUTDIR}/$JOBNAME.$NODE.$JOBID.lst $PVE $EXTRA $NCLIENTS $ME $CPARMS &
	CRC=$?

	if [[ $CRC -ne 0 ]]; then
	    echo "Unable to start clients, rc=$CRC"
	    return
	fi

    done

}

# ------------------------------------------------ RunQCon()

function RunQCon() {

    export MAPREDUCE_PID_FILE=`mktemp -t MRPIDXXX`

    trap '' SIGHUP

    if [[ "$NTREES" -ne "" ]]; then
	PNT="-t $NTREES"
    fi

    if [[ "$ME" = "othello" ]]; then
	export QCON_Map_Mem_Threshold=4000000000
	export QCON_Reduce_Interm_Write=30000000
    else if [[ `hostname -f` = *brazos.tamu.edu ]]; then
	export QCON_Map_Mem_Threshold=1500000000
	export QCON_Reduce_Interm_Write=30000000
    fi; fi

    . $PVE

    $PRE $QC -n $NSRVCLIENTS $PNT -w $WDBNAME -d $RDBNAME -c $JOBNAME $TREEFILE $SPARMS
    QRC=$?
    if [[ $QRC -ne 0 ]]; then 
	echo "QCon abnormal terminationq:, rc=$QRC" 
    fi

}

# ------------------------------------------------ RunTest()

function RunTest() {
    
    echo "************************ QCon Starting on node $ME at `date +'%D %T.%N'`"

#    StartDB

    StartClients
    if [[ $CRC -ne 0 ]]; then
	StopDB
	exit 1
    fi

    RunQCon

    if [[ $QRC -eq 0 ]]; then
#	UnloadDB
	DeleteWorkDB
    fi

#    StopDB

    wait

    echo "************************ QCon Ending on node $ME at `date +'%D %T.%N'`"

}
