__precompile__()
module MINLPLib

using JuMP
using JSON 
using Glob

include("features.jl")

METAATTRS = ["LIBRARY", "NAME",
            "NVARS",
            "NBINVARS", "NINTVARS", "NNLVARS",
            "NCONS",
            "NLINCONS", "NNLCONS",
            "OBJSENSE", "OBJTYPE", "NLOPERANDS",
            "STATUS",
            "OBJVAL", "OBJBOUND",
            "SOURCE"]

PROTECTED_LIBS = []

minlplib_dir = joinpath(dirname(pathof(MINLPLib)), "..")

special_instances = String[]

export fetch_model

end #module
