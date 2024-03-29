"""
Import the Julia environment, specifically the ``Metacells.jl`` package.
"""

from daf import jl

jl.seval("using Metacells")
