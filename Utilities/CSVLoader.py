from __future__ import print_function

"""
Load a csv file into an array of objects.
Attributes will be named according to the csv header line
"""

__author__ = """
Ralph W. Crosby
rwc@cs.tamu.edu
Department of Computer Science and Engineering
Texas A&M University
College Station, TX 77843
"""

__all__ = ['CSVLine', 'CSVLoader', 'DupColumnError', 'ParseFileNameMetaclass']

# *************************************************************************


import csv
import re
import sys

# *************************************************************************

class ParseFileNameMetaclass(type):
    """
    Metaclass to allow for class property definition
    """
    
    @property
    def attrs(self):
        return list(self._attrs)

# *************************************************************************

class DupColumnError(object):
    """
    Exception object generated with a duplicate column is detected
    """
    
    def __init__(self, cname):
        self.__columnName = cname

    def __str__(self):
        return "Duplicate column name {}".format(self.__columnName)

# *************************************************************************

class CSVLine(object):
    """
    Representation of a line in the csv file with attributes named by the 
    columns of the file.
    """

    # *****************************************************************
    
    def __init__(self, *args):
        """
        Initialize the line optionally with the set of attributes provided
        """
        
        if len(args):
            [self.MakeAttr(k,v) for k,v in args[0].items()]

    # *****************************************************************

    def MakeAttr(self, name, val):
        
        try:
            v = int(val)
        except ValueError:
            try:
                v = float(val)
            except ValueError:
                if val == "True":
                    v = True
                elif val == "False":
                    v = False
                else:
                    v = val

        setattr(self, name, v)

# *************************************************************************

class CSVLoader(list):
    """
    Collection of csv lines
    """

    # *****************************************************************

    def __init__(self, files, fnParser=None, verbose=False, encoding=None):

        self.attrs = ['Filename']
        self.originalAttrs = {}
        self.fnParser = None
        self.encoding = encoding

        if fnParser:
            self.fnParser = fnParser
            self.attrs += fnParser.attrs

        if type(files) is list:
            [self.AddFile(f, verbose=verbose) for f in files]
        else:
            self.AddFile(files, verbose=verbose)

    # *****************************************************************

    def AddFile(self, file, verbose=False):
        """
        Add a file's data to the collection
        """
        
        fp = open(file, 'r', encoding=self.encoding) if type(file) is str else file

        if verbose:
            print("Loading input file", fp.name, file=sys.stderr)

        fnAttrs = self.__ParseFileName(fp.name)

        [self.attrs.append(a) for a in fnAttrs if a not in self.attrs]

        fileAttrs = []                        # Attributes for current file
            
        for row in csv.reader(fp):            # Loop through lines in the file

            if not len(fileAttrs):            # Handle the header line

                for rawAttr in row:
                    attr = _MakeAttr(rawAttr)
                    if attr in fileAttrs:
                        raise DupColumnError(attr)
                    fileAttrs.append(attr)
                    self.originalAttrs[attr] = rawAttr
                        
                    if verbose:
                        print("Attribute: /{}/".format(attr), file= sys.stderr)

                continue

            line = CSVLine(fnAttrs)           # Make a new test instance
            [line.MakeAttr(fileAttrs[i], row[i]) for i in range(len(row))]
            self.append(line)                 # Append to this list

        # Add any new attributes to the overall list
        [self.attrs.append(a) for a in fileAttrs if a not in self.attrs]

    # *****************************************************************
    
    def __ParseFileName(self, fn):
        """
        Parse the file name generating attribute values for 
        all records from this file
        """

        fnAttrs = {'Filename' : fn}

        if self.fnParser:
            attrs = self.fnParser.Parse(fn)
            [fnAttrs.update([(k, _TryNumeric(v))]) for k, v in attrs]

        return fnAttrs

# *************************************************************************

def _MakeAttr(a):
    """
    Remove spaces, special characters from attribute names
    """
    a = re.sub('%', 'Pct', a)                     # Convert percent sign
    return re.sub('\W', '', a)                    # Remove special characters

# *************************************************************************
      
def _TryNumeric(val):
    """
    Try to make the value numeric
    """
        
    try:
        v = int(val)
    except ValueError:
        try:
            v = float(val)
        except ValueError:
            v = val

    return v

# *************************************************************************

if __name__ == '__main__':
    sys.exit("Not a main program")
