# Copyright (c) 2018: Matthew Wilhelm & Matthew Stuber.
# This code is licensed under MIT license (see LICENSE.md for full details)
#############################################################################
# McCormick.jl
# A McCormick operator library in Julia
# See https://github.com/PSORLab/McCormick.jl
#############################################################################
# src/implicit_routines/contact.jl
# Definitions for contract functions used in implicit relaxations.
#############################################################################

"""
$(SIGNATURES)

Performs a single step of the parametric method associated with `t` assumes that
the inputs have been preconditioned.
"""
function contract! end

"""
$(TYPEDSIGNATURES)

Applies the Gauss-Siedel variant of the Newton type contractor.
"""
function contract!(t::NewtonGS, d::MCCallback{FH,FJ,C,PRE,N,T}) where {FH <: Function,
                                                                       FJ <: Function,
                                                                       C, PRE, N,
                                                                       T<:RelaxTag}
    S = zero(MC{N,T})
    @. d.x0_mc = d.x_mc
    for i = 1:d.nx
        S = zero(MC{N,T})
        for j = 1:d.nx
            if (i !== j)
                @inbounds S += d.J[i,j]*(d.x_mc[j] - d.z_mc[j])
            end
        end
        @inbounds d.x_mc[i] = d.z_mc[i] - (d.H[i] + S)*McCormick.inv1(d.J[i,i], 1.0/d.J[i,i].Intv)
        @inbounds d.x_mc[i] = final_cut(d.x_mc[i], d.x0_mc[i])
    end
    return
end

"""
$(TYPEDSIGNATURES)

Applies the componentwise variant of the Krawczyk type contractor.
"""
function contract!(t::KrawczykCW, d::MCCallback{FH,FJ,C,PRE,N,T}) where {FH <: Function,
                                                                         FJ <: Function,
                                                                         C, PRE, N,
                                                                         T<:RelaxTag}
    S::MC{N,T} = zero(MC{N,T})
    @. d.x0_mc = d.x_mc
    for i=1:d.nx
        S = zero(MC{N,T})
        for j=1:d.nx
            if (i !== j)
                @inbounds S -= (d.J[i,j])*(d.x_mc[j] - d.z_mc[j])
            else
                @inbounds S += (one(MC{N,T}) - d.J[i,j])*(d.x_mc[j] - d.z_mc[j])
            end
        end
        @inbounds d.x_mc[i] = d.z_mc[i] - d.H[i] + S
        @inbounds d.x_mc[i] = final_cut(d.x_mc[i], d.x0_mc[i])
    end
    return
end
