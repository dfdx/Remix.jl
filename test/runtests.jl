using Test
using Remix
import FiniteDifferences as FD


rand_cotangent(::Real) = randn()
rand_cotangent(x::AbstractArray) = convert(typeof(x), randn(size(x)...))


function test_vjp(f, args...; atol=1e-5, rtol=1e-5)
    # AD
    val, res = vjp_fwd(f, args...)
    dy = rand_cotangent(val)
    ad_dxs = vjp_bwd(f, res, dy)
    # FD
    fdm = FD.central_fdm(5, 1)
    fd_dxs = FD.j′vp(fdm, f, dy, args...)

    @test all(isapprox.(ad_dxs, fd_dxs))
end


@testset "arraymath" begin
    test_vjp(*, 2.0, 3.0)
    test_vjp(*, rand(2, 3), rand(3, 2))
end