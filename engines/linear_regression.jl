linreg(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:AbstractFloat} = [ones(length(x)) x]\y
function add_linear_reg!(ax, X, Y)
    b, a = linreg(X, Y)
    lines!(ax,[minimum(X), maximum(X)], [minimum(X) * a + b, maximum(X) * a + b],linewidth = 5, color  =:red, linestyle =:dash)
    return a
end 