# EXPERIMENTAL: alternative tracer based on operator overloading


const CURRENT_TAPE = Ref{Any}(Tape(Umlaut.BaseCtx()))


current_tape() = CURRENT_TAPE[]
function current_tape!(tape::Tape)
    old_tape = CURRENT_TAPE[]
    CURRENT_TAPE[] = tape
    return old_tape
end


mutable struct ShapedArray{T, N}
    shape::NTuple
    var::Variable
end

Base.size(x::ShapedArray) = x.shape
Base.show(io::IO, x::ShapedArray{T, N}) where {T, N} = print(io, "ShapedArray{$T, $N}($(x.var), $(x.shape))")
Base.show(io::IO, ::MIME"text/plain", x::ShapedArray{T, N}) where {T, N} = print(io, "ShapedArray{$T, $N}($(x.var), shape=$(x.shape))")


ShapedArray(x::AbstractArray{T, N}) where {T, N} = ShapedArray{T, N}(size(x), V(0))





function Base.:*(x::ShapedArray{T, 2}, y::ShapedArray{T, 2}) where T
    tape = current_tape()
    val = ShapedArray{T,2}((x.shape[1], y.shape[2]), V(0))
    v = push!(tape, mkcall(*, x.var, y.var; val=val))
    val.var = v
    return val
end

macro tracked(ex)
end


# @shaped *(x::ShapedArray{T, 2}, y::ShapedArray{T, 2}) where T = ShapedArray()


macro shaped(ex)
    @assert Meta.isexpr(ex, :(=))
    sig, out_expr = ex.args
    body = quote
        tape = current_tape()
        out = $out_expr
        v = push!(tape, mkcall(*, x.var, y.var; val=out))
        out.var = v
        return out
    end

    return Expr(:function, sig, Expr(:block, body.args...))

end


function trace_shaped(f, args...; ctx=Umlaut.BaseCtx())
    tape = Tape(ctx)
    old_tape = current_tape!(tape)
    s_args = []
    for arg in args
        arg isa AbstractArray || error("Can only trace a function with all-array arguments")
        s_arg = ShapedArray(arg)
        v = push!(tape, Input(s_arg))
        s_arg.var = v
        push!(s_args, s_arg)
    end
    res = f(s_args...)
    tape.result = res.var
    current_tape!(old_tape)
    return res, tape
end



foo(x, y) = x * y

function main()
    f = foo
    args = (rand(2, 3), rand(3, 4))
    ctx = Umlaut.BaseCtx()
    trace_shaped(f, args...; ctx=ctx)
end


function main2()
    sig = :(Base.:*(x::ShapedArray{T, 2}, y::ShapedArray{T, 2}) where T)


    ex = :(foo(x::ShapedArray, y::ShapedArray) = ShapedArray(x.shape[1], y.shape[2]))
end