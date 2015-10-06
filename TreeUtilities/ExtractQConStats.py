"""
Extract various statistics from QCon runs.
"""

__author__ = """
Ralph W. Crosby
rwc@cs.tamu.edu
Department of Computer Science and Engineering
Texas A&M University
College Station, TX 77843
"""

__all__ = ['QConRun', 'Statistics', 'BadStatError']

# *************************************************************************

import re
import sys

# *************************************************************************

class BadStatError(BaseException):
    """
    Exception object used when a required statistic is missing
    """
    
    def __init__(self, fn, stat):
        self.fn = fn
        self.stat = stat

    def __str__(self, ):
        return "Problem with statistic {} in file {}".format(self.stat, self.fn)

# *************************************************************************

class QConRun(object):
    """
    A set of statistics associated with a single QCon run
    """

    # *****************************************************************
    
    def __init__(self):
        self.server = False

    # *****************************************************************

    def _PQConEnd(self, m):
        """
        Indicate that this was a server run
        """
        self.server = True

    # *****************************************************************
    
    def _PExit(self, m):
        """
        Process the exit status line, validate object
        """
        pass

    # *****************************************************************
    
    def ParseFileName(self, fn, fnParser):
        """
        Parse the file name passed into attributes
        """
        self.filename = fn
        attrs = fnParser.Parse(fn)
        [setattr(self, k, _TryNumeric(v)) for k, v in attrs]

    # *****************************************************************

    @property
    def CSVLine(self):
        """
        Return a line of csv data for the run
        """
    	
        l = []
        for k in sorted(Statistics.attrs.keys()):
            l.append(self.__dict__[k])

        return ','.join(str(i) for i in l)

# *************************************************************************
        
class Statistics(list):
    """
    A set of extracted QQRun objects
    """

    # *****************************************************************
    
    class _Stat(object):
        """
        Object for a particular statistic
        """
        
        def __init__(self, rexp):
            self.re = re.compile(rexp)

    # *****************************************************************
    
    def __init__(self, fnParser, files=None, verbose=False):

        self.fnParser = fnParser
        Statistics.attrs.update(fnParser.attrs)
        [self.ProcessFile(f, verbose) for f in files]

    # *****************************************************************
    
    def ProcessFile(self, fn, verbose=False): 

        with open(fn, 'r') as inf:

            if verbose:
                print("Processing {:s}".format(fn), file=sys.stderr)

            runObj = QConRun()

            runObj.ParseFileName(fn, self.fnParser)

            for line in inf:
            
                for a in Statistics._re:
                    result = a.re.match(line)
                    if result:
                        if hasattr(a, 'attr'):
                            [setattr(runObj, 
                                     a.attr[i], 
                                     _TryNumeric(result.group(i+1))) 
                             for i in range(len(a.attr))]
                        else:
                            a.func(runObj, result)
                        break

        if runObj.server:
            del runObj.server
            self.append(runObj)

    # *****************************************************************
    
    @property
    def CSVHeader(self):
        """
        Output a string containing the header for each csv line
        """
    	
        return ','.join('"{}"'.format(Statistics.attrs[i][1]) 
                        for i in sorted(Statistics.attrs.keys()))
       
# *************************************************************************
      
def _TryNumeric(val):
    """
    Try to make the value numeric
    """
        
    try:
        v = int(val.replace(',',''))
    except ValueError:
        try:
            v = float(val)
        except ValueError:
            v = val

    return v

# *************************************************************************

if __name__ == '__main__':
    sys.exit("Not a main program")

# Setup the attributes        

__reTable = [
    ('\[.*\]\s+Database Server.*\:\s+(\S+)',                                 ['dataServer']),
    ('\[.*\]\s+Database Name.*\:\s+(\S+)',                                   ['dataName']),
    ('\[.*\]\s+Collection Prefix.*\:\s+(\S+)',                               ['collPrefix']),
    ('\[.*\]\s+Trees Processed.*\:\s+([\d,]+)',                              ['treesProcessed']),
    ('\[.*\]\s+Total Unique Quartets.*\:\s+([\d,]+)',                        ['uniqueQuartets']),
    ('\[.*\]\s+Singular Quartets.*\:\s+([\d,]+)',                            ['singularQuartets']),
    ('\[.*\]\s+Quartets in Majority.*\:\s+([\d,]+)',                         ['majorityQuartets']),
    ('\[.*\]\s+Quartets in All Trees.*\:\s+([\d,]+)',                        ['strictQuartets']),
    ('\s+Elapsed \(wall clock\) time.*\:\s+(\d+:\d+:\d+|\d+:\d+\.\d+)', ['elapsedTime']),
    ('\s+Exit status',                                                       QConRun._PExit),
    ('\[.*\].*QCon.py ended normally',                                       QConRun._PQConEnd),

]
            
Statistics._re = []
for sin in __reTable:
    sobj = Statistics._Stat(sin[0])
    if type(sin[1]) is list:
        sobj.attr = sin[1]
    else:
        sobj.func = sin[1]
    Statistics._re.append(sobj)

Statistics.attrs = {
    'dataServer' :           ('Database Server',   'Data Server'),
    'dataName' :             ('Database Name',     'Data Name'),
    'collPrefix' :           ('Collection Prefix', 'Coll Prefix'),
    'treesProcessed' :       ('Trees Processed',   'Trees Proc'),
    'uniqueQuartets' :       ('Unique Quartets',   'Unique'),
    'singularQuartets' :     ('Singular Quartets', 'Singular'),
    'majorityQuartets' :     ('Majority Quartets', 'Majority'),
    'strictQuartets' :       ('Strict Quartets',   'Strict'),
    'elapsedTime' :          ('Elapsed Time',      'Elapsed'),
    'filename' :             ('Filename',          'Filename'),

}
