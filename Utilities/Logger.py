"""
Standard logging configuration
"""

__author__ = """
Ralph W. Crosby
rwc@cs.tamu.edu
Department of Computer Science and Engineering
Texas A&M University
College Station, TX 77843
"""

__all__ = ['Logger', 'ElapsedTime', 'FMT_SHORT', 'FMT_LONG']

# *************************************************************************

import logging
import sys
from   time     import time

# *************************************************************************

class ElapsedTime:
    """ Add the elapsed time to the logging output"""

    starttime = time()

    def filter(self, record):
        record.etime = time() - ElapsedTime.starttime
        return True

# *************************************************************************

FMT_SHORT = '[%(asctime)s.%(msecs)03d %(etime)06.3f %(levelname)-8s] %(message)s'
FMT_LONG  = '[%(asctime)s.%(msecs)03d %(etime)06.3f %(name)-9s %(levelname)-8s] %(message)s'

def Logger(stream=sys.stderr,
           level=logging.INFO,
           name=None,
           fmt=FMT_SHORT):

    logging.basicConfig(format=fmt,
                        datefmt='%H:%M:%S', 
                        stream=stream)


    log = logging.getLogger(name if name else None)
    log.setLevel(level)
    log.addFilter(ElapsedTime())

    return log

# *************************************************************************

if __name__ == '__main__':
    sys.exit("Not a main program")
