vjp_fwd(::typeof(*), x::Number, y::Number) = x * y, (x, y)
vjp_bwd(::typeof(*), res::Tuple{<:Number, <:Number}, g) = (g * res[2], g * res[1])

vjp_fwd(::typeof(*), x::AbstractMatrix, y::AbstractMatrix) = x * y, (x, y)
vjp_bwd(::typeof(*), res, g) = (g * res[2]', res[1]' * g)


vjp_fwd(::typeof(+), x::Number, y::Number) = (x + y, ())
vjp_bwd(::typeof(+), res, g) = (g, g)


