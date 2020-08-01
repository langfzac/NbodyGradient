abstract type PreAllocArrays end
abstract type AbstractDerivatives{T} <: PreAllocArrays end

mutable struct Jacobian{T<:AbstractFloat} <: AbstractDerivatives{T}
    jac_phi 
    jac_kick
    jac_copy
    jac_ij
    jac_tmp1 
    jac_tmp2
    jac_err1
    dqdt_ij
    dqdt_phi
    dqdt_kick

    function Jacobian(::Type{T},n::Integer) where T<:AbstractFloat
        sevn::Int64 = 7*n
        return new{T}([SizedMatrix{sevn,sevn}(zeros(T,sevn,sevn)) for _ in 1:3]...,
                      SizedMatrix{14,14}(zeros(T,14,14)),
                      [SizedMatrix{14,sevn}(zeros(14,sevn)) for _ in 1:3]...,
                      SizedVector{14}(zeros(T,14)),
                      [SizedVector{sevn}(zeros(T,sevn)) for _ in 1:2]...)
    end
end

mutable struct dTime{T<:AbstractFloat} <: AbstractDerivatives{T}

    jac_phi
    jac_kick
    jac_ij
    dqdt_phi
    dqdt_kick
    dqdt_ij
    dqdt_tmp1
    dqdt_tmp2

    function dTime(::Type{T},n::Integer) where T<:AbstractFloat
        sevn::Int64 = 7*n
        return new{T}([SizedMatrix{sevn,sevn}(zeros(T,sevn,sevn)) for _ in 1:2]...,
                      SizedMatrix{14,14}(zeros(T,14,14)),
                      [SizedVector{sevn}(zeros(T,sevn)) for _ in 1:2]...,
                      SizedVector{14}(zeros(T,14)),
                      [SizedVector{14}(zeros(T,14)) for _ in 1:2]...)
    end
end

