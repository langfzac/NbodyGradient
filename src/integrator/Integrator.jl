import NbodyGradient: InitialConditions

#========== Integrator ==========#
abstract type AbstractIntegrator end

"""

Integrator to be used, and relevant parameters.
"""
struct Integrator{T<:AbstractFloat} <: AbstractIntegrator
    scheme::Function
    h::T
    t0::T
    tmax::T

    Integrator(scheme::Function,h::T,t0::T,tmax::T) where T<:AbstractFloat = new{T}(scheme,h,t0,tmax)
end

Integrator(scheme::Function,h::Real,t0::Real,tmax::Real) = Integrator(scheme,promote(h,t0,tmax)...)

# Default to ah18!
Integrator(h::T,t0::T,tmax::T) where T<:AbstractFloat = Integrator(ah18!,h,t0,tmax)
Integrator(h::Real,t0::Real,tmax::Real) = Integrator(ah18!,promote(h,t0,tmax)...)

#========== State ==========#
abstract type AbstractState end

"""

Current state of simulation.
"""
struct State{T<:AbstractFloat} <: AbstractState
    x
    v
    t::T
    m
    jac_step
    dqdt
    terror::T
    xerror # These might be put into an 'PreAllocArrays'
    verror
    jac_error
    n::Int64
end

"""Constructor for State. Uses initial conditions struct."""
function State(ic::InitialConditions{T}) where T<:AbstractFloat
    x,v,_ = init_nbody(ic)
    sx1,sx2 = size(x)
    sv1,sv2 = size(v)
    n = ic.nbody
    terror = zero(T) 
    xerror = SizedMatrix{sx1,sx2}(zeros(T,sx1,sx2))
    verror = SizedMatrix{sv1,sv2}(zeros(T,sv1,sv2))
    jac_step = SizedMatrix{7*n,7*n}(Matrix{T}(I,7*n,7*n))
    dqdt = SizedVector{7*n}(zeros(T,7*n))
    jac_error = SizedMatrix{7*n,7*n}(zeros(T,7*n,7*n))
    return State{T}(x,v,ic.t0,ic.m,jac_step,dqdt,terror,xerror,verror,jac_error,ic.nbody)
end

"""Steps the time of state with compenstated summation"""
function step(s::State{T},h) where T<:AbstractFloat
    t,terr = comp_sum(s.t,s.terror,h)
    return State(s.x,s.v,t,s.m,s.jac_step,s.dqdt,terr,s.xerror,s.verror,s.jac_error,s.n)
end

"""Shows if the positions, velocities, and Jacobian are finite."""
Base.show(io::IO,::MIME"text/plain",s::State{T}) where {T} = begin
    println(io,"State{$T}:"); 
    println(io,"Positions  : ", all(isfinite.(s.x)) ? "finite" : "infinite!"); 
    println(io,"Velocities : ", all(isfinite.(s.v)) ? "finite" : "infinite!");
    println(io,"Jacobian   : ", all(isfinite.(s.jac_step)) ? "finite" : "infinite!");
    return
end 

#========== Running Methods ==========#
"""

Callable `Integrator` method. Integrates to `i.tmax`.
"""
function (i::Integrator)(s::State{T}) where T<:AbstractFloat 

    # Preallocate struct of arrays for derivatives (and pair)
    d = Jacobian(T,s.n) 
    pair = zeros(Bool,s.n,s.n)

    while s.t < (i.t0 + i.tmax)
        # Take integration step and advance time
        i.scheme(s,d,i.h,pair)
        s = step(s,i.h)
    end
    return
end

#========== Includes  ==========#
const ints = ["ah18"]
for i in ints; include(joinpath(i,"$i.jl")); end

const ints_no_grad = ["ah18","dh17"]
for i in ints_no_grad; include(joinpath(i,"$(i)_no_grad.jl")); end
