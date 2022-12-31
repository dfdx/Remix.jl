vjp_fwd(::typeof(*), x::Number, y::Number) = x * y, ()
vjp_bwd(res, g, ::typeof(*), x::Number, y::Number) = (g * y, g * x)

vjp_fwd(::typeof(*), x::AbstractMatrix, y::AbstractMatrix) = x * y, (x, y)
vjp_bwd(res, g, ::typeof(*), x::AbstractMatrix, y::AbstractMatrix) = (g * y', x' * g)


vjp_fwd(::typeof(+), x::Number, y::Number) = (x + y, ())
vjp_bwd(res, g, ::typeof(+), x::Number, y::Number) = (g, g)

# TODO: rewrite to consist only of primitives
_unsum(x, dy, dims) = broadcast(last∘tuple, x, dy)
_unsum(x, dy, ::Colon) = broadcast(last∘tuple, x, Ref(dy))


vjp_fwd(::typeof(sum), x::AbstractArray; dims=:) = sum(x, dims=dims), (x,)
vjp_bwd(res, g, ::typeof(sum)) = _unsum(x, g, dims)
