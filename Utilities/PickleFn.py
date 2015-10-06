"""
Python module to allow pickling of simple and closure functions
"""

__author__ = """
Ralph W. Crosby
rwc@cs.tamu.edu
Department of Computer Science and Engineering
Texas A&M University
College Station, TX 77843
"""

__all__ = ['Fn2Tuple', 'Tuple2Fn']

# *************************************************************************

import marshal
import sys
import types

# **************************************************
        
def Fn2Tuple(fn):
    """
    Create a serialized version of a function including
    any closure values required
    """
    fnCode = marshal.dumps(fn.__code__)
    fnClosure = [v.cell_contents for v in fn.__closure__] if fn.__closure__ else None

    return (fnCode, fn.__name__, fn.__defaults__, fnClosure)

# **************************************************
        
def Tuple2Fn(tpl):
    """
    Create a function including closure information
    from an unpickled tuple
    """
    cstr, name, defaults, closurel = tpl

    code = marshal.loads(cstr)

    def MakeCell(v):
        return (lambda: v).__closure__[0]

    closure = tuple([MakeCell(v) for v in closurel]) if closurel else None

    return types.FunctionType(code, globals(), name, defaults, closure)

# *************************************************************************

if __name__ == '__main__':
    sys.exit("Not a main program")
