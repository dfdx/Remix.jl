using Test
using Remix
import Remix.Umlaut
import FiniteDifferences as FD


rand_cotangent(::Real) = randn()
rand_cotangent(x::AbstractArray) = convert(typeof(x), randn(size(x)...))


function test_vjp(f, args...; atol=1e-5, rtol=1e-5)
    # AD
    val, res = vjp_fwd(f, args...)
    dy = rand_cotangent(val)
    ad_dxs = vjp_bwd(res, dy, f, args...)
    # FD
    fdm = FD.central_fdm(5, 1)
    fd_dxs = FD.j′vp(fdm, f, dy, args...)

    @test all(isapprox.(ad_dxs, fd_dxs; atol=atol, rtol=rtol))
end


@testset "arraymath" begin
    test_vjp(*, 2.0, 3.0)
    test_vjp(*, rand(2, 3), rand(3, 2))
end



@testset "tapeutils" begin
    f = (x, y) -> sum(x * y; dims=1)
    args = (rand(2, 3), rand(3, 2))
    _, tape = Umlaut.trace(f, args...; ctx=RemixCtx())
    short_tape = Remix.remove_unused(tape)
    @test length(short_tape) == 5
    # test that the result points to the same object in memory, i.e. no data copy
    @test tape.result.op.val === short_tape.result.op.val

    Umlaut.play!(short_tape, f, args...)
    # test that the result has the same value
    @test tape.result.op.val == short_tape.result.op.val
end