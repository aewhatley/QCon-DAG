#!/usr/bin/env python
"""
Combine a set of graphics files into a single pdf.
"""

__author__ = """
Ralph W. Crosby
rwc@cs.tamu.edu
Department of Computer Science and Engineering
Texas A&M University
College Station, TX 77843
"""

# **************************************************

from   jinja2 import Environment, FileSystemLoader
import os
import sys

# **************************************************

def ParseCommandLine():

    import argparse

    p = argparse.ArgumentParser(description = __doc__)

    p.add_argument('files', 
                   nargs='+',
                   help='Input graphics files',
                   default="")

    p.add_argument('-o', '--output',
                   help="Output .tex file",
                   type=argparse.FileType('w'),
                   required=True)

    p.add_argument('-t', '--template',
                   help="Jinja template file",
                   type=str,
                   required=True)

    return p.parse_args()

# **************************************************

if __name__ == '__main__':

    opt  = ParseCommandLine()                     

    env = Environment(loader=FileSystemLoader(['/', '.']),
                      variable_start_string='[[',
                      variable_end_string=']]')

    tex = env.get_template(opt.template)

    print(tex.render(files=opt.files,
                     title=os.getcwd()),
          file=opt.output)
