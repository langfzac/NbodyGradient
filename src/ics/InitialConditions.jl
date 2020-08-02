import StaticArrays: StaticArrayLike

abstract type AbstractInitialConditions end
"""

Holds orbital elements of a single body.
"""
struct Elements{T<:AbstractFloat} <: AbstractInitialConditions
    m::T
    P::T
    t0::T
    ecosϖ::T
    esinϖ::T
    I::T
    Ω::T
    e::T
    ϖ::T
end

function Elements(;m=0.0,P=0.0,t0=0.0,e=0.0,ϖ=0.0,I=0.0,Ω=0.0)
    esinϖ,ecosϖ = e.*sincos(ϖ)
    return Elements(m,P,t0,ecosϖ,esinϖ,I,Ω,e,ϖ)
end

function Base.show(io::IO, ::MIME"text/plain" ,elems::Elements{T}) where T <: Real
    names = ["m","P","t0","ecosϖ","esinϖ","I","Ω","e","ϖ"]
    vals = [elems.m,elems.P,elems.t0,elems.ecosϖ,elems.esinϖ,elems.I,
            elems.Ω,elems.e,elems.ϖ]
    println(io, "Elements{$T}")
    println.(Ref(io), names,": ",vals)
end

#= Fix this later...
"""Allow alternate specification"""
function Elements(;m::T=0.0,P::T=0.0,t0::T=0.0,e::T=0.0,ϖ::T=0.0,i::T=0.0,Ω::T=0.0) where T<:Real
    esinϖ,ecosϖ = e .* sincos(ϖ)
    return new{T}(m,P,t0,ecosϖ,esinϖ,i,Ω)
end

"""Allow keywargs"""
Elements(;m::T=0.0,P::T=0.0,t0::T=0.0,ecosϖ::T=0.0,esinϖ::T=0.0,i::T=0.0,Ω::T=0.0) where T<:Real = Elements(m,P,t0,ecosϖ,esinϖ,i,Ω)
=#

#= This overwrites the main constructor... Maybe not needed. I don't see folks using integers here.
"""Promotion to (atleast) floats"""
Elements(m::Real=0.0,P::Real=0.0,t0::Real=0.0,e::Real=0.0,ϖ::Real=0.0,i::Real=0.0,Ω::Real=0.0) = Elements(promote(m,P,t0,e,ϖ,i,Ω)...)
=#

abstract type InitialConditions{T} end
"""

Holds relevant initial conditions arrays. Uses orbital elements.
"""
struct ElementsIC{T<:AbstractFloat} <: InitialConditions{T}
    elements::StaticArrayLike{T}
    H::StaticArrayLike{Int64}
    ϵ::StaticArrayLike{T}
    amat::StaticArrayLike{T}
    nbody::Int64
    m::StaticArrayLike{T}
    t0::T
    der::Bool

    function ElementsIC(elems::Union{String,Matrix{T},StaticArrayLike{T}},H::Union{Array{Int64,1},Int64},t0::T;der::Bool=true) where T <: AbstractFloat
        # Check if only number of bodies was passed. Assumes fully nested.
        typeof(H) == Int64 ? H = @SVector([H,ones(H-1)...]) : H = SVector{size(H)...}(H)
        n = H[1]
        ϵ = SizedMatrix{n,n,T}(hierarchy(H))
        elements = change_elems_type(elems,n)
        m = SizedVector{n}(elements[1:n,1])
        amat = amatrix(ϵ,m)
        return new{T}(elements,H,ϵ,amat,n,m,t0,der)
    end
end

    
# Deals with different elements types in ElementsIC
function change_elems_type(elems::String,n); return SizedMatrix{n,7}(readdlm(elems,',',comments=true)); end
function change_elems_type(elems::Matrix,n); return SizedMatrix{n,7}(elems); end
function change_elems_type(elems::StaticArrayLike,n); return elems; end

"""

Collects `Elements` and produces an `ElementsIC` struct.
"""
function ElementsIC(t0::T,elems::Elements{T}...;H::Vector{Int64}) where T <: AbstractFloat
    
    elements = zeros(length(elems),7)
    function parse_system(elems::Elements{T}...) where T <: AbstractFloat
        bodies = Dict{Symbol,Elements}()
        for i in 1:length(elems)
            key = Meta.parse("b$i")
            bodies[key] = elems[i]
        end
        key = sort(collect(keys(bodies)))
        fields = setdiff(fieldnames(Elements),(:e,:ϖ))
        for i in 1:length(bodies), elm in enumerate(fields)
            elements[i,elm[1]] = getfield(bodies[key[i]],elm[2])
        end
        return elements
    end     
            
    elements .= parse_system(elems...)
    return ElementsIC(elements,H,t0)
end

"""Shows the elements array."""
Base.show(io::IO,::MIME"text/plain",ic::ElementsIC{T}) where {T} = begin
println(io,"ElementsIC{$T}\nOribital Elements: "); show(io,"text/plain",ic.elements); end;

"""

Holds relevant initial conditions arrays. Uses Cartesian coordinates. 
"""
struct CartesianIC{T<:AbstractFloat} <: InitialConditions{T}
    x::Array{T,2}
    v::Array{T,2}
    jac_init::Array{T,2}
    m::Array{T,1}
    t0::T
    nbody::Int64
    
    function CartesianIC(filename::String,x,v,jac_init,t0) where T <: AbstractFloat
        m = convert(Array{T},readdlm(filename,',',comments=true))
        nbody = length(m)
        return new{T}(x,v,jac_init,m,t0,nbody)
    end
end

# Include ics source files
const ics = ["kepler","kepler_init","setup_hierarchy","init_nbody"]
for i in ics; include("$(i).jl"); end
