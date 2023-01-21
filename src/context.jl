###############################################################################
#                                  Context                                    #
###############################################################################

struct RemixCtx
    # in the mappings below keys are unboud variables in source tape
    # values are bound variables in target tape
    outs::Dict{V, V}        # V(src_v.id) => vjp_fwd(...)[1]
    residuals::Dict{V, V}   # V(src_v.id) => vjp_fwd(...)[2]
    derivs::Dict{V, V}      # V(src_v.id) => g
end

RemixCtx() = RemixCtx(Dict(), Dict(), Dict())
Base.show(io::IO, ctx::RemixCtx) =
    print(io, "RemixCtx($(length(ctx.outs)) vars, $(length(ctx.derivs)) derivs)")


function Umlaut.isprimitive(::RemixCtx, f, args...)
    @nospecialize
    is_base = Umlaut.isprimitive(Umlaut.BaseCtx(), f, args...)
    is_appl = applicable(vjp_fwd, f, args...)
    if is_base && !is_appl
        t_args_str = join(["::$(typeof(a))" for a in args], ", ")
        throw(AssertionError(
            "Function \n\n\t$f($t_args_str)\n\nis not a Remix primitive " *
            "and cannot be traced further since it's defined in one of " *
            "the Julia's base modules. Define vjp_fwd() and vjp_bwd() for it " *
            "or mark it using @constcall\n"
        ))
    end
    return is_appl
    # F = Core.Typeof(f)
    # Args = Core.Typeof.(args)
    # return Core.Compiler.return_type(vjp_fwd, Tuple{F, Args...}) !== Nothing
end


###############################################################################
#                                  constcall                                  #
###############################################################################


macro constcall(sig)
    fn, args... = sig.args
    arg_names = [Meta.isexpr(a, :(::)) ? a.args[1] : a for a in args]
    return quote
        # TODO
        Umlaut.isprimitive(::RemixCtx, ::typeof($fn), $(args...)) = true

        function Umlaut.record_primitive!(tape::Tape{RemixCtx}, v_fn::typeof($fn), $(args...))
            fn = v_fn isa V ? tape[v_fn].val : v_fn
            args = Umlaut.var_values([$(arg_names...)])
            val = fn(args...)
            return push!(tape, Constant(val))
        end
    end
end


# constexpr(fn) = fn()

# """
#     @constexpr <expression>

# During tracing, don't analyze the expression but only evaluate and record
# the result as a Constant operation.

# During normal execution is equivalent to `identity(<expression>)`
# """
# macro constexpr(ex)
#     if Meta.isexpr(ex, :call)
#         return esc(:(constexpr(() -> $ex)))
#     else
#         throw(AssertionError("@constexpr's argument must be a call, but got: $ex"))
#     end
# end

# Umlaut.isprimitive(::RemixCtx, ::typeof(constexpr), fn) = true

# function Umlaut.record_primitive!(tape::Tape{RemixCtx}, ::typeof(constexpr), v_fn)
#     fn = v_fn isa V ? tape[v_fn].val : v_fn
#     val = fn()
#     return push!(tape, Constant(val))
# end



# ###

# function constexpr_example(x)
#     return @constexpr size(repeat(x, outer=(2, 2)))
# end