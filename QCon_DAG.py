#!/usr/bin/env python
"""
Compute the quartet consensus set using Map-Reduce.

Environment Variables (map):

QCON_Map_Save_Threshold: Minimum quartets associated with a key to save
QCON_Map_Mem_Threshold: Max quartet save memory
QCON_Map_Check_Freq: How often (in quartets generated) to check the memory utilization

Environment Variables (reduce):

QCON_Reduce_Mem_Threshold: Max quartet save memory
QCON_Reduce_Check_Freq: How often (in quartets generated) to check the memory utilization

"""
from CombineTrees2 import ReadIn

__author__ = """
Ralph W. Crosby
rwc@cs.tamu.edu
Department of Computer Science and Engineering
Texas A&M University
College Station, TX 77843
"""

# **************************************************
# Copyright (c) 2013 Ralph W. Crosby, Texas A&M University, College Station, Texas
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# **************************************************

import argparse
import logging
from   pprint    import pprint      as pp
import pymongo
import re
import os
import socket
import sys
  
import TreeUtilities.NewickParser as Parser
from TreeUtilities.NewickParser import Parser as Parser1
from   TreeUtilities.Tree         import *
import Utilities.Logger           as Logger
import Utilities.MapReduce        as MapReduce
from   Utilities.PickleFn         import Fn2Tuple, Tuple2Fn
from TreeUtilities.Tree           import DAG
from DAGQuartets                  import DAGQuartets
from CombineTrees2                import ReadIn  
import time
import pympler


# **************************************************

class Bunch(object):
    """Collect a bunch of variables"""
    
    def __init__(self, **stuff):
        self.__dict__.update(stuff)

# **************************************************

DEFAULT_MONGOURI = "mongodb://localhost"
DEFAULT_DB = "Quartets" 
DEFAULT_LOGLEVEL = "info"
DEFAULT_TREELIMIT = 0

class Options(object):
    """
    Parse command line options
    """

    def __init__(self, desc):

        p = argparse.ArgumentParser(description = desc,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

        p.add_argument('files', 
                       nargs='+', 
                       type=argparse.FileType('r'),
                       help='List of newick tree files')

        p.add_argument('-t', '--treelimit',
                       help="Limit number of trees to process (0=all)",
                       type=int,
                       default=DEFAULT_TREELIMIT)

        p.add_argument('-m', '--mongouri',
                       help="URI for the mongo database",
                       default=DEFAULT_MONGOURI)

        p.add_argument('-d', '--db',
                       help="Database name",
                       default=DEFAULT_DB)

        p.add_argument('-w', '--workdb',
                       help="Work Database name",
                       default=DEFAULT_DB)

        p.add_argument('-c', '--collectionprefix',
                       help="Prefix to use for all collections",
                       default=None)

        p.add_argument('-n', '--nclients',
                       help='Number of clients to run locally (omitted, use num cpus)',
                       type=int)

        loggerValues = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

        p.add_argument('-l', '--loglevel',
                       help='Logger level (debug, info, warning, error, critical)',
                       type=lambda x : x.upper(),
                       choices=loggerValues,
                       default=DEFAULT_LOGLEVEL)

        p.parse_args(namespace=self)

# *************************************************************************

class TreeReader(object):
    """
    Generator class to return newick strings
    """

    def __init__(self, files, limit, log, collstats):
        self.limit = limit
        self.log   = log
        self.stats = collstats
        self.files = files if type(files) is list else [files]

    def __iter__(self):

        myDAG = DAG()
        myDAG.Combine(ReadIn(self.files))

        for f in self.files:        
            fp = open(f, 'r') if type(f) is str else f
            fp.seek(0)
            self.tno = 0
            for tree in Parser.Reader(fp, ';'):
                ttree = tree.strip()
                if len(ttree):
                    self.log.debug("Tree {}: {}...".format(self.tno, ttree[:20]))
                    yield ("{} {}".format(fp.name, self.tno), (ttree + ';', self.tno, myDAG))
                    self.tno += 1
                if self.limit and self.tno == self.limit:
                   break
            else:
                fp.close()
                continue
            fp.close()
            break

        self.stats.insert(dict(trees=self.tno))
        
        self.log.info('Read {} trees from input'.format(self.tno))


#***************************************************
def MapFn(dbinfo, **cargs):
    """
    Map function closure
    Facilitates passing the database parameters to the clients.
    """
    MAX_DB_RCD             = 15000000               # Limit for a mongo record (actually slightly less)

    ENV_SAVE_THRESHOLD     = r'QCON_Map_Save_Threshold' # Min quartets for a key to save
    DEFAULT_SAVE_THRESHOLD = 10000                      # 10k

    ENV_MEM_THRESHOLD      = r'QCON_Map_Mem_Threshold'  # Memory use limitation
    DEFAULT_MEM_THRESHOLD  = 1000000000                 # Approx 1GB memory limit

    ENV_CHECK_FREQ         = r'QCON_Map_Check_Freq'     # How often (in quartets) to check memory
    DEFAULT_CHECK_FREQ     = 50000000                   # Every 50M quartets

    # **************************************************

    def Fn(key, value, logger, **kwargs):
        """
        Parse newick string and emit quartets
        """
        # **************************************************

        def LoadQuartet(k1 ,k2, k3, k4, num):
            """
            Load a value into the quartets structure updating the length
            to match the generated BSON length
            """

            def ValidChars(str, search=re.compile(r'[^a-zA-Z0-9]').search):
                return not bool(search(str))

            FmtKey = lambda k : "'{}':{}".format(k, ':'.join('{:02x}'.format(ord(c)) for c in k))

            if not ValidChars(k1) or not ValidChars(k2) or not ValidChars(k3) or not ValidChars(k4):
                logger.warning("Invalid characters in quartet {},{}|{},{}".format(FmtKey(k1), 
                                                                                  FmtKey(k2), 
                                                                                  FmtKey(k3), 
                                                                                  FmtKey(k4)))
                k1 = re.sub(r'[^a-zA-Z0-9]', r'?', k1)
                k2 = re.sub(r'[^a-zA-Z0-9]', r'?', k2)
                k3 = re.sub(r'[^a-zA-Z0-9]', r'?', k3)
                k4 = re.sub(r'[^a-zA-Z0-9]', r'?', k4)

            '''if k1 not in quartets:
                quartets[k1] = dict(length = 64 + len(k1) + len(k2) + len(k3) + len(k4), # each element in quartets now has a term value, the number of times the quartet appears
                             data = {k2 : { k3 : { k4 : num } } },
                             nQ = num)
            elif k2 not in quartets[k1]['data']:
                quartets[k1]['data'][k2] = { k3 : {k4 : num} }
                quartets[k1]['length'] += 26 + len(k2) + len(k3) + len(k4)
                quartets[k1]['nQ'] += num
            elif k3 not in quartets[k1]['data'][k2]:
                quartets[k1]['data'][k2][k3] = {k4 : num}
                quartets[k1]['length'] += 19 + len(k3) + len(k4)
                quartets[k1]['nQ'] += num
            elif k4 not in quartets[k1]['data'][k2][k3]:
                quartets[k1]['data'][k2][k3][k4] = num
                quartets[k1]['length'] += 7 + len(k4) + int(log10(len(quartets[k1]['data'][k2][k3]) - 1) + 1)
                quartets[k1]['nQ'] += num
            else:
                quartets[k1]['data'][k2][k3][k4] += num  
                quartets[k1]['nQ'] += num
            
                if quartets[k1]['length'] >= MAX_DB_RCD:
                    collWork.insert(dict(key=k1, data=quartets[k1]['data']))
                    logger.debug("Wrote for taxa /{}/, {:,} quartets".format(k1, quartets[k1]['nQ']))
                    del quartets[k1]'''
            # there are issues with creating dictionary entries in quartets right now (hashing dictionaries entries is apparently very slow)
            if k1 not in quartets:
                quartets[k1] = dict(length = 128 + len(k1) + len(k2) + len(k3) + len(k4) + int(log2(num)), 
                             data = {k2 : { k3 : [k4, num] }},
                             nQ = num)
            elif k2 not in quartets[k1]['data']:
                quartets[k1]['data'][k2] = { k3 : [k4, num] }
                quartets[k1]['length'] += 52 + len(k2) + len(k3) + len(k4) + int(log2(num))
                quartets[k1]['nQ'] += num
            elif k3 not in quartets[k1]['data'][k2]:
                quartets[k1]['data'][k2][k3] = [k4, num]
                quartets[k1]['length'] += 38 + len(k3) + len(k4) + int(log2(num))
                quartets[k1]['nQ'] += num
            else:
                quartets[k1]['data'][k2][k3].append(k4)
                quartets[k1]['data'][k2][k3].append(num)
                quartets[k1]['length'] += 14 + len(k4) + int(log10(len(quartets[k1]['data'][k2][k3]) - 1) + 1) + int(log2(num))
                quartets[k1]['nQ'] += num
                
                if quartets[k1]['length'] >= MAX_DB_RCD:
                    collWork.insert(dict(key=k1, data=quartets[k1]['data']))
                    logger.debug("Wrote for taxa /{}/, {:,} quartets".format(k1, quartets[k1]['nQ']))
                    del quartets[k1]
        # **************************************************
        
        def SaveAll(threshold=None):
            """
            Save all records to the database and clear the quartets structure
            """

            logger.debug("Starting SaveAll")
            insList = [dict(key=k, data=v['data']) for k, v in quartets.items() 
                       if not threshold or v['nQ'] >= threshold]
            collWork.insert(insList)
            for k in sorted(quartets.keys()):
                if not threshold or quartets[k]['nQ'] >= threshold:
                    logger.debug("{:11,d} quartets for taxa '{}'".format(quartets[k]['nQ'], k))
                    del quartets[k]
            logger.debug("Finished SaveAll")

        # **************************************************
        
        def WriteTreeStats(tree):
            """
            Write information about the current tree to the stats collection
            """

            collStats.insert(dict(treeid=key,
                                  ntaxa=tree.nLeaves,
                                  taxa=[l.label for l in tree.leaves],
                                  nquartets=tree.nQuartets,
                                  diameter=tree.diameter))
 
        # **************************************************

        # The imports need to be here so they get done in the client processes

        from   pymongo         import MongoClient
        from   math            import log10, log2
        import os
        from   pympler.asizeof import asizeof
        from   random          import uniform
        import re
        import asyncio
        import time

        from TreeUtilities.NewickParser import Parser
        from TreeUtilities.Quartets     import Quartets
        from TreeUtilities.Tree         import DAG
        from DAGQuartets                import DAGQuartets
        from CombineTrees2              import ReadIn
        
        # Get environment parameters for the reduce
        
        saveThreshold = int(os.getenv(ENV_SAVE_THRESHOLD, DEFAULT_SAVE_THRESHOLD))
        memThreshold  = int(os.getenv(ENV_MEM_THRESHOLD, DEFAULT_MEM_THRESHOLD))
        memCheckFreq  = int(os.getenv(ENV_CHECK_FREQ, DEFAULT_CHECK_FREQ))

        logger.info("Save Threshold  : {:,d}".format(saveThreshold))
        logger.info("Memory Threshold: {:,d}".format(memThreshold))
        logger.info("Check Frequency : {:,d}".format(memCheckFreq))

        # Get mongo collection 

        DBInfo=Tuple2Fn(dbinfo)
        collWork, collResults, collStats, conn = DBInfo(**cargs)
        logger.info("DB connected: {} id {}".format(conn['client'], conn['connectionId']))

        tree = next((t for t in Parser(value[0])))
        myDAG = value[2]
        allQuartets = DAGQuartets(myDAG)
        starting_nodes = myDAG.top()[value[1]]

        WriteTreeStats(tree)

        # Loop accumulating the quartets 
        nQ = 0 
        quartets = {}
        for q in allQuartets.MultiprocessDFS(starting_nodes): # value[3] is allQuartets
            LoadQuartet(q[0][0],q[0][1],q[1][0],q[1][1],q[2])
            nQ += q[2]

            if nQ % memCheckFreq == 0:
                mem=asizeof(quartets)
                if mem > memThreshold:
                    logger.info("Saving at {:,} quartets using {:,} bytes of memory".format(nQ, mem))
                    SaveAll(saveThreshold)
                    quartets={}
        # Output remaining records to the database
        SaveAll()

        # Return the set of taxa labels as the result set

        for taxa in tree.leaves:
           yield taxa.label, 1
            
        logger.info("Produced {:,} quartets".format(nQ))

    return Fn

# **************************************************

def ReduceFn(dbinfo, **cargs): # reduce phase
    """
    Reduce function closure
    Facilitates passing the database parameters to the clients.
    """

    MAX_DB_RCD            = 15000000              # Limit for a mongo record (actually slightly less)
    DEFAULT_RETRIES       = 5                     # Retries on cursor failure
    INTERMEDIATE_BATCH_SZ = 50                    # Records to batch insert into intermediate collection
    RESULTS_BATCH_SZ      = 10                    # Records to batch insert into results collection

    ENV_INTERM_WRITE      = r'QCON_Reduce_Interm_Write'  # How often (in quartets) to write intermediate
    DEFAULT_INTERM_WRITE  = 100000                       # Write intermediate after 100k unique quartets

    def Fn(tx1, values, logger, **kwargs):
        """
        Accumulate values for a key (left side of the quartets)
        """

        # The imports need to be here so they get done in the client processes

        from   collections     import defaultdict
        import datetime
        from   math            import log10
        import os
        import pymongo
        from   pympler.asizeof import asizeof
        from   time            import perf_counter

        fmtTime = lambda p, t: "{:25s}: {!s}".format(p, datetime.timedelta(seconds=(t)))

        # *************************************************************************
    
        class TimedCursor(object):

            def __init__(self, mkfn):
                start = perf_counter()
                self.cursor = mkfn()
                self.totaltime = perf_counter() - start

            def __iter__(self):
                return self

            def __next__(self):
                start = perf_counter()
                rcd = next(self.cursor)
                self.totaltime += perf_counter() - start
                return rcd
            
        # **************************************************
        
        def AccumQuartets():
            """
            Process work table accumulating quartets
            Returns the intermediate MongoDB collection if one was
            generated
            """

            nonlocal intermSaveTime

            startTime = perf_counter()
            memCheckTime = 0.0

            collInterm = None                     # Intermediate Mongo collection
            lastId = None                         # Last record id processed
            retries = 0                           # Current retry count
            nUniqueQ = 0                          # Quartets processed this group
            tUniqueQ = 0                          # Total unique quartets

            quartetDef = lambda : defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

            cursor = TimedCursor(lambda : collWork.find(dict(key=tx1), dict(data=True)).sort('_id', 1))
            quartets = quartetDef()

            while True:
                try:

                    for data1 in cursor: 
                        lastId = data1['_id']
                        retries = 0
                        for tx2, data2 in data1['data'].items():
                            for tx3, data3 in data2.items():
                                '''for tx4, value in data3.items(): # originally for tx4, value in data3.items():
                                    if tx4 not in quartets[tx2][tx3]:
                                        nUniqueQ += 1 
                                    quartets[tx2][tx3][tx4[0]] += value'''
                                for i in range(0, int(len(data3)/2)):
                                    if data3[2*i] not in quartets[tx2][tx3]:
                                        nUniqueQ += 1
                                    quartets[tx2][tx3][data3[2*i]] += data3[2*i+1]
                                                                                      

                        if nUniqueQ >= intermWrite:
                            logger.info("{:25s}: {:,} quartets, {:,} bytes".format("Writing Intermediate", nUniqueQ, asizeof(quartets)))
                            if not collInterm:
                                collInterm = MakeIntermCollection()
                            intermSaveTime += BatchWriter(collInterm, 
                                                          MakeIntermediateRecord(quartets), 
                                                          INTERMEDIATE_BATCH_SZ)
                            quartets = quartetDef()
                            tUniqueQ += nUniqueQ
                            nUniqueQ = 0

                    break

                except pymongo.errors.OperationFailure as excp:
                
                    if retries >= DEFAULT_RETRIES:
                        logger.error("Unable to obtain data after {} retries, failing".format(retries))
                        raise 

                    logger.info("Operation failure: {}".format(str(excp)))

                    retries += 1
                    saveTime = cursor.totaltime
                    cursor = TimedCursor(lambda :collWork.find({'key' : tx1, 
                                                                '_id' : {'$gt' : lastId } }, 
                                                               dict(data=True)).sort('_id', 1))
                    cursor.totaltime += saveTime
                                         
            logger.info(fmtTime("Work Fetch Time", cursor.totaltime))

            if collInterm:
                if len(quartets):
                    logger.info("{:25s}: {:,} quartets, {:,} bytes".format("Writing Intermediate", nUniqueQ, asizeof(quartets)))
                    intermSaveTime += BatchWriter(collInterm, 
                                                  MakeIntermediateRecord(quartets), 
                                                  INTERMEDIATE_BATCH_SZ)
                logger.info(fmtTime("Intermediate Insert Time", intermSaveTime))
                

            tUniqueQ += nUniqueQ
            logger.info("{:25s}: {:14,d}".format("Unique Quartets Found", tUniqueQ))
            logger.info(fmtTime("Phase 1 Time", perf_counter() - startTime))
            logger.info(fmtTime("Phase 1 Time", perf_counter() - startTime))

            return collInterm, quartets

        # **************************************************
        
        def BatchWriter(coll, iterator, limit):
            """
            Write batches of records from the passed iterator to the passed collection
            returns time required for the operation
            """
            totalTime = 0.0
            batch = []
            for rcd in iterator:

                batch.append(rcd)

                if len(batch) >= limit:
                    startTime = perf_counter()
                    coll.insert(batch)
                    totalTime += perf_counter() - startTime
                    batch = []

            if len(batch):
                startTime = perf_counter()
                coll.insert(batch)
                totalTime += perf_counter() - startTime

            return totalTime

        # **************************************************
        
        def IncrCounts(qcnt):
            nonlocal uniqueQ
            nonlocal singularQ
            nonlocal majorityQ
            nonlocal strictQ

            uniqueQ += 1
            # take into account that each quartet is double-counted
            if qcnt == 2:
                singularQ += 1
            if qcnt > 2*tnoMajority:
                majorityQ += 1
            if qcnt == 2*tnoStrict:
                strictQ +=1 

        # **************************************************
    
        def MakeIntermCollection():
            """
            Generate the intermediate collection
            """

            ic = collWork.database["{}_{!s}".format(collWork.name, tx1)]
            ic.drop()
            ic.ensure_index([('tx2', pymongo.ASCENDING),
                             ('tx3', pymongo.ASCENDING),
                             ('_id', pymongo.ASCENDING)])
            logger.debug("Created intermediate table: {}".format(ic.name))
            return ic

        # **************************************************
    
        def MakeIntermediateRecord(quartets):
            """
            Build and return intermediate records
            """

            for tx2, v2 in quartets.items():
                for tx3, v3 in v2.items():
                    rcd = dict(tx2 = tx2, tx3 = tx3, tx4list = [])
                    for tx4, qcount in v3.items():
                        rcd['tx4list'].append(dict(tx4 = tx4, count=qcount))
                    yield rcd

        # **************************************************
        
        def MakeResultsRecord(quartets):
            """
            Build a results record
            """

            for tx2, v2 in quartets.items():
                rcd, rcdLen = ResetResultsRcd(tx2)

                for tx3, v3 in v2.items():
                    for tx4, qcnt in v3.items():

                        IncrCounts(qcnt)
                        rcd['rhslist'].append(dict(rhs=[tx3, tx4], 
                                                   count=qcnt))
                        rcdLen += 48 + len(tx3) + len(tx4) + int(log10(len(rcd['rhslist'])) + 1)

                        if rcdLen >= MAX_DB_RCD:
                            logger.info("Wrote early results for [{},{}] len {}".format(tx1, tx2, rcdLen))
                            yield rcd
                            rcd, rcdLen = ResetResultsRcd(tx2)

                yield rcd

                quartets[tx2] = None

        # **************************************************
        
        def MakeResultsFromInterm(collInterm):
            """
            Create a results records from the intermediate collection
            """

            def UpdateRecord():
                """
                Update the results record from queued interm data
                """

                nonlocal tx4Counts
                nonlocal rcd
                nonlocal rcdLen

                def DoUpdate(tx4, qcnt):
                    nonlocal rcd
                    nonlocal rcdLen
                    IncrCounts(qcnt)
                    rcd['rhslist'].append(dict(rhs=[saveTx3, tx4], count=qcnt))
                    rcdLen += 48 + len(row['tx3']) + len(tx4) + \
                              int(log10(len(rcd['rhslist'])) + 1)

                if len(tx4Counts):
                    [DoUpdate(tx4, qcnt) for tx4, qcnt in tx4Counts.items()]
                    tx4Counts = defaultdict(int)

            cursor = TimedCursor(lambda :collInterm.find().sort([('tx2', pymongo.ASCENDING),
                                                                 ('tx3', pymongo.ASCENDING),
                                                                 ('_id', pymongo.ASCENDING)]))

            saveTx2 = None
            saveTx3 = None
            saveId  = None
            tx4Counts = defaultdict(int)
            rcd = None
            retries = 0                           # Current retry count

            while True:
                try:

                    for row in cursor:
                        if row['tx2'] != saveTx2:
                            UpdateRecord()
                            if rcd:
                                yield rcd
                            saveTx2 = row['tx2']
                            saveTx3 = None
                            rcd, rcdLen = ResetResultsRcd(row['tx2'])

                        if row['tx3'] != saveTx3:
                            UpdateRecord()
                            if rcdLen >= MAX_DB_RCD:
                                logger.info("Wrote early results for [{},{}] len {}".format(tx1, row['tx2'], rcdLen))
                                yield rcd
                                rcd, rcdLen = ResetResultsRcd(row['tx2'])
                            saveTx3 = row['tx3']

                        for tx4 in row['tx4list']:
                            tx4Counts[tx4['tx4']] += tx4['count']

                        saveId = row['_id']

                    break

                except pymongo.errors.OperationFailure as excp:
                
                    if retries >= DEFAULT_RETRIES:
                        logger.error("Unable to obtain data after {} retries, failing".format(retries))
                        raise 

                    logger.info("Operation failure: {}".format(str(excp)))

                    retries += 1
                    saveTime = cursor.totaltime
                    cursor = TimedCursor(lambda :collInterm.find({'tx2' : {'$gte' : saveTx2 }, 
                                                                  'tx3' : {'$gte' : saveTx3 },
                                                                  '_id' : {'$gt' : saveId } } ).sort([('tx2', pymongo.ASCENDING),
                                                                                                      ('tx3', pymongo.ASCENDING),
                                                                                                      ('_id', pymongo.ASCENDING)]))
                    cursor.totaltime += saveTime

            if rcd:
                UpdateRecord()
                yield rcd

            logger.info(fmtTime("Intermediate Fetch Time", cursor.totaltime))

        # **************************************************

        def ResetResultsRcd(tx2):
            """
            Return an empty, initialzed results record and base length
            """

            return dict(lhs=[tx1, tx2], rhslist=[]), 62 + len(tx1) + len(tx2)

        # **************************************************

        reduceStart = perf_counter()

        # Get environment variables

        intermWrite  = int(os.getenv(ENV_INTERM_WRITE, DEFAULT_INTERM_WRITE))

        logger.info("{:25s}: {:14,d}".format("Intermediate Write Count", intermWrite))

        # Database interface initialization

        DBInfo=Tuple2Fn(dbinfo)
        collWork, collResults, collStats, conn = DBInfo(**cargs)
        logger.info("{:25s}: {} id {}".format("DB connected", conn['client'], conn['connectionId']))

        # Phase 1 - Collect the quartets for this key

        intermSaveTime = 0.0                      # Total time to save to intermediate collection

        collInterm, quartets = AccumQuartets()

        p1End = perf_counter()

        # Get the number of trees being processed

        tcd = collStats.find_one(dict(trees={'$exists' : True}))
        tnoMajority = int(tcd['trees'] / 2)       # 50% of the trees
        tnoStrict   = tcd['trees']                # All trees

        # Initialize counts

        uniqueQ = 0                               # Number of unique quartets
        majorityQ = 0                             # Number of quartets in 50% of the trees
        strictQ = 0                               # Number of quartets in all trees
        singularQ = 0                             # Number of quartets in only one tree

        qMem = asizeof(quartets)
        logger.info("{:25s}: {:14,d}".format("Final Accum Memory", qMem))
        
        p2Start = perf_counter()
        logger.info(fmtTime("Phase 1-2 Time", p2Start - p1End))

        # Phase 2 - Write the quartets from either the collected list or the intermediate table

        workInsTime = BatchWriter(collResults, 
                                  MakeResultsFromInterm(collInterm) if collInterm else MakeResultsRecord(quartets), 
                                  RESULTS_BATCH_SZ)
        p2End = perf_counter()

        logger.info(fmtTime("Work Insert Time", workInsTime))
        logger.info(fmtTime("Phase 2 Time", p2End - p2Start))
        logger.info(fmtTime("Reduce Total Time", p2End - reduceStart))

        return (uniqueQ, majorityQ, strictQ, singularQ, len(values))

    return Fn

# **************************************************

def DBInfo(collectionprefix, mongouri, db, workdb, localHostInUri, serverFQDN, uriFormat):
    """
    Return collection objects for the three collections:
    _Work    : temp collection used to pass quartets from map to reduce phases
    _Results : where the quartets will end up
    _Stats   : Relevant statistics about the run

    If localhost is in the uri:
    Determine if we're on the local host
    otherwise replace with true host fqdn

    """

    from pymongo import MongoClient

    if localHostInUri:
        import socket
        me = socket.getfqdn()
        if me != serverFQDN:
            mongouri = uriFormat.format(serverFQDN)

    conn = MongoClient(mongouri)
    db = conn[db]
    wdb = conn[workdb]

    inprog = db['$cmd.sys.inprog']
    data=inprog.find_one({'$all' : True})

    me = db.command('whatsmyuri')

    for connection in data['inprog']:
        if 'client' in connection and connection['client'] == me['you']:
            break
    else:
        raise LookupError("Unable to find current connection information")
        
    return wdb["{}_Work".format(collectionprefix)], \
        db["{}_Results".format(collectionprefix)], \
        db["{}_Stats".format(collectionprefix)], \
        connection

# **************************************************

def SetupDatabaseOptions(uri, db, workdb, cprefix, cname, tl):
    """
    Parse and format the database options into a format the map and
    reduce functions can use
    """

    lhre = re.compile("localhost", re.IGNORECASE)
    (uriFormat, localHostInUri) = lhre.subn('{}', uri)

    if not cprefix:
        cprefix = "QCon_{}_{}".format(os.path.basename(cname).replace('-', '_'),
                                      tl)

    return dict(collectionprefix = cprefix,
                mongouri         = uri,
                db               = db,
                workdb           = workdb,
                localHostInUri   = True if localHostInUri else False,
                serverFQDN       = socket.getfqdn(),
                uriFormat        = uriFormat)

# **************************************************

def Run(files,
        mongouri=DEFAULT_MONGOURI,
        db=DEFAULT_DB,
        workdb=DEFAULT_DB,
        collectionprefix=None,
        treelimit=DEFAULT_TREELIMIT,
        nclients=None,
        loglevel=DEFAULT_LOGLEVEL,
        log=None):
    """
    Do the actual run
    the code is structured this way to make setting up automated testing easier
    """

    loglevel = loglevel.upper()
    if not log:
        log = Logger.Logger(fmt=Logger.FMT_LONG, level=loglevel)

    # Setup database connection
    
    #myDAG = DAG()
    #myDAG.Combine(ReadIn(files))
    #allQuartets = DAGQuartets(myDAG)

    firstNameFn = lambda f : f if type(f) is str \
                  else (f[0] if type(f[0]) is str else f[0].name) if type(f) is list \
                  else f.name

    dbopts = SetupDatabaseOptions(mongouri, db, workdb, collectionprefix, firstNameFn(files), treelimit)
    collWork, collResults, collStats, conn = DBInfo(**dbopts)

    collWork.remove()
    collWork.ensure_index([('key', pymongo.ASCENDING),
                           ('_id', pymongo.ASCENDING)])
    collResults.remove()
    collStats.remove()

    # Create closures for the map and reduce functions

    dbinfo = Fn2Tuple(DBInfo)
    #mapFn = MapFn(dbinfo=dbinfo, myDAG = myDAG, allQuartets = allQuartets, **dbopts)
    mapFn = MapFn(dbinfo=dbinfo, **dbopts)
 
    reduceFn = ReduceFn(dbinfo=dbinfo, 
                        **dbopts)

    # Do it

    datasource = TreeReader(files,
                            treelimit,
                            log,
                            collStats)

    server = MapReduce.Server(datasource,
                              mapFn,
                              reduceFn,
                              password="QCon",
                              nclients=nclients,
                              loglevel=loglevel)
    try:
        results = server()
    except MapReduce.MapReduceError as mre:
        return None

    # Update the number of trees record with the stats

    rlist = list(results.values())
    stats = dict(zip(['uniqueQ', 'majorityQ', 'strictQ', 'singularQ'],
                     [sum(x[y] for x in rlist) for y in range(len(rlist[0]))]))



    tcd = collStats.find_one(dict(trees={'$exists' : True}))
    stats.update(tcd)           # Add in the number of trees
    collStats.save(stats)       # Write back to the database

    stats['dbserver'] = dbopts['serverFQDN']
    stats['db'] = dbopts['db']
    stats['colprefix'] = dbopts['collectionprefix']
    del stats['_id']            # Don't want the mongo object id
    return stats

#    collWork.remove()

# **************************************************

if __name__ == '__main__':

    opt = Options(__doc__)
    log = Logger.Logger(fmt=Logger.FMT_LONG, level=opt.loglevel)
    log.info("Starting {}".format(sys.argv[0]))
    # Show options
    log.info(__doc__.splitlines()[1])
    log.info("Options specified:")
    for o in sorted(opt.__dict__.keys()):
        log.info("{:12s} = {}".format(o, opt.__dict__[o]))


    stats = Run(log=log, **vars(opt))

    # Log the stats

    if stats:
        log.info("Database Server       : {:s}".format(stats['dbserver']))
        log.info("Database Name         : {:s}".format(stats['db']))
        log.info("Collection Prefix     : {:s}".format(stats['colprefix']))
        log.info("Trees Processed       : {:15,d}".format(stats['trees']))
        log.info("Total Unique Quartets : {:15,d}".format(stats['uniqueQ']))
        log.info("Singular Quartets     : {:15,d}".format(stats['singularQ']))
        log.info("Quartets in Majority  : {:15,d}".format(stats['majorityQ']))
        log.info("Quartets in All Trees : {:15,d}".format(stats['strictQ']))

        log.info('{} ended normally'.format(sys.argv[0]))
        sys.exit(0)
    else:
        log.info('{} ended abnormally'.format(sys.argv[0]))
        sys.exit(1)
