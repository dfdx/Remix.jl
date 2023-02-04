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
    # global ff = f
    # global aargs = args
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
end


function Umlaut.record_primitive!(tape::Tape, v_fargs...)
    line = get(tape.meta, :line, nothing)
    # fold constants
    v_fargs = [v isa V && tape[v] isa Constant ? tape[v].val : v for v in v_fargs]
    if any(v -> v isa V, v_fargs)
        push!(tape, mkcall(v_fargs...; line=line))
    else
        f, args... = v_fargs
        push!(tape, Constant(f(args...)))
    end
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