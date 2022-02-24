
const USE_MIN_RANGE = true

struct AffineEAGO{N}
    c::Float64               # mid-point
    γ::SVector{N,Float64}    # affine terms
    Δ::Float64               # error term
end

function AffineEAGO(x::AffineEAGO{N}, p::Float64, q::Float64, δ::Float64) where N
    c = p * x.c + q
    γ = p .* x.γ
    Δ = p * x.Δ + δ
    AffineEAGO{N}(c, γ, δ)
end
mid(x::Interval{Float64}) = 0.5*(x.lo + x.hi)
AffineEAGO{N}(x::Float64, X::Interval{Float64}, i::Int) where N = AffineEAGO{N}(mid(X), radius(X)*seed_gradient(i, Val(N)), 0.0)

const UNIT_INTERVAL = Interval{Float64}(-1,1)
Interval(x::AffineEAGO{N}) where N = x.c + x.Δ*UNIT_INTERVAL + sum(y -> abs(y)*UNIT_INTERVAL, x.γ)
function bounds(x::AffineEAGO{N}) where N
    z = Interval(x)
    z.lo, z.hi
end

zero(::Type{AffineEAGO{N}}) where N = AffineEAGO{N}(0.0, zero(SVector{N,Float64}), 0.0)
zero(x::AffineEAGO{N}) where N = AffineEAGO{N}(0.0, zero(SVector{N,Float64}), 0.0)

one(::Type{AffineEAGO{N}}) where N = AffineEAGO{N}(1.0, zero(SVector{N,Float64}), 0.0)
one(x::AffineEAGO{N}) where N = AffineEAGO{N}(1.0, zero(SVector{N,Float64}), 0.0)

+(x::AffineEAGO{N}, y::AffineEAGO{N}) where N = AffineEAGO{N}(x.c + y.c, x.γ .+ y.γ, x.Δ + y.Δ)
+(x::AffineEAGO{N}, α::Float64) where N = AffineEAGO{N}(α+x.c, x.γ, x.Δ)
+(α::Float64, x::AffineEAGO{N}) where N = x + α

function *(x::AffineEAGO{N}, y::AffineEAGO{N}) where N
    c = x.c * y.c
    γ = x.c .* y.γ + y.c .* x.γ
    pos_sum = 0.0
    neg_sum = 0.0
    for i = 1:N
        vt = y.γ[i]*x.γ[i]
        if vt > 0.0
            pos_sum += vt
        else
            neg_sum -= vt
        end 
    end
    Δ = max(pos_sum, neg_sum)
    for i = 1:N
        xi = x.γ[i]
        yi = y.γ[i]
        for j = 1:(i-1)
            Δ += yi*x.γ[j] + y.γ[j]*xi
        end
    end
    return AffineEAGO{N}(c, γ, Δ)
end
*(x::AffineEAGO{N}, α::Float64) where N = AffineEAGO{N}(α*x.c, α.*x.γ, abs(α)*x.Δ)
*(α::Float64, x::AffineEAGO{N}) where N = x*α

function -(x::AffineEAGO{N}, y::AffineEAGO{N}) where N
    AffineEAGO{N}(x.c - y.c, x.γ .- y.γ, x.Δ + y.Δ)
end
-(x::AffineEAGO{N}) where N = AffineEAGO{N}(-x.c, .-(x.γ), x.Δ)
-(x::AffineEAGO{N}, α::Float64) where N = AffineEAGO{N}(x.c - α, x.γ, x.Δ)
-(α::Float64, x::AffineEAGO{N}) where N = α + (-x)

/(x::AffineEAGO{N}, α::Float64) where N = AffineEAGO{N}(x.c/α, x.γ/α, x.Δ/abs(α))
/(α::Float64, x::AffineEAGO{N}) where N = α*inv(x)

function log(x::AffineEAGO{N}) where N
    a, b = bounds(x)
    fa = log(a)
    fb = log(b)
    if USE_MIN_RANGE
        p = 1/b
        q = 0.5*(fa + fb - p*(a + b))
        Δ = abs(0.5*(fb - fa - p*(b - a)))
        return AffineEAGO(x, p, q, Δ)
    end
    p = (fb - fa)/(b - a)
    ξ = 1/p
    fξ = log(ξ)
    q = 0.5*(fa + fξ - p*(a + ξ))
    Δ = abs(0.5*(fξ - fa - p*(ξ - a)))
    return AffineEAGO(x, p, q, Δ)
end

function log10(x::AffineEAGO{N}) where N
    a, b = bounds(x)
    fa = log10(a)
    fb = log10(b)
    if USE_MIN_RANGE
        p = 1/(b*log(10))
        q = 0.5*(fa + fb - p*(a + b))
        Δ = abs(0.5*(fb - fa - p*(b - a)))
        return AffineEAGO(x, p, q, Δ)
    end
    p = (fb - fa)/(b - a)
    ξ = 1/p
    fξ = log10(ξ)
    q = 0.5*(fa + fξ - p*(a + ξ))
    Δ = abs(0.5*(fξ - fa - p*(ξ - a)))
    return AffineEAGO(x, p, q, Δ)
end

function pow_1d(x::AffineEAGO{N}, n::Number, p) where N
    a, b = bounds(x)
    fa = a^n
    fb = b^n
    if USE_MIN_RANGE
        q = 0.5*(fa + fb - p*(a + b))
        Δ = abs(0.5*(fb - fa - p*(b - a)))
        return AffineEAGO(x, p, q, Δ)
    end
    p = (fb - fa)/(b - a)
    ξ = (p/n)^(1/(n - 1))
    fξ = ξ^n
    q = 0.5*(fa + fξ - p*(a + ξ))
    Δ = abs(0.5*(fξ - fa - p*(ξ - a)))
    return AffineEAGO(x, p, q, Δ)
end

function pow_even(x::AffineEAGO{N}, n::Int) where N
    a, b = bounds(x)
    fa = a^n
    fb = b^n
    if USE_MIN_RANGE
        m = min(0.0, a, b)
        M = max(0.0, a, b)
        p = 0.0
        q = 0.5*(m + M)
        Δ = 0.5*(M - m)
        return AffineEAGO(x, p, q, Δ)
    end
    p = (fb - fa)/(b - a)
    ξ = (p/n)^(1/(n - 1))
    fξ = ξ^n
    q = 0.5*(fa + fξ - p*(a + ξ))
    Δ = abs(0.5*(fξ - fa - p*(ξ - a)))
    return AffineEAGO(x, p, q, Δ)
end

function pow_odd(x::AffineEAGO{N}, n::Int) where N
    a, b = bounds(x)
    fa = a^n
    fb = b^n
    p = (fb - fa)/(b - a)
    q = 0.0
    ξ = (p/n)^(1/(n-1))
    fξ = ξ^n
    Δ = abs(fξ - p*ξ)
    return AffineEAGO(x, p, q, Δ)
end

function ^(x::AffineEAGO{N}, n::Int) where N

    iszero(n) && zero(x)
    isone(n)  && one(x)

    xL, xU = bounds(x)
    if (xL > 0.0) || (xU < 0.0)
        return pow_1d(x, n, (n >= 0) ? n*xL^(n-1) : n*xU^(n-1))
    elseif iseven(n)
        return pow_even(x, n)
    end
    (xL < 0.0 < xU) && error("Undefined domain...")
    return pow_odd(x, n)
end

function ^(x::AffineEAGO{N}, n::Float64) where N
    if isinteger(n)
        return x^Int(n)
    end
    xL, xU = bounds(x)
    (xL < 0.0) && error("Invalid domain...")
    if (n > 1.0) || (n < 0.0)
        return pow_1d(x, n, n*xU^(n-1))
    end
    return pow_1d(x, n, n*xL^(n-1))
end

# DONE...
function ^(x::AffineEAGO{N}, n::Number) where N
    if iszero(n) 
        return zero(x)
    elseif isone(n)
        return one(x)
    end
    return x^n
end

function inv(x::AffineEAGO{N}) where N
    a, b = bounds(x)
    (a < 0.0 < b) && error("Invalid domain...")
    if b < 0.0
        return -inv(-x)
    end
    if USE_MIN_RANGE
        p = -1/b^2
        q = -(p*(a + b)^2)/(2*a)
        Δ = -(p*(a - b)^2)/(2*a)
        return AffineEAGO(x, p, q, Δ)
    end
    p = -1/(a*b)
    q = -0.5*p*(sqrt(a) + sqrt(b))^2
    Δ = -0.5*p*(sqrt(a) - sqrt(b))^2
    return AffineEAGO(x, p, q, Δ)
end

struct MCAffPnt{N,T}
    v::MC{N,T}
    box::AffineEAGO{N}
end
MC(x::MCAffPnt{N,T}) where {N,T<:RelaxTag} = x.v
MC(x::MC{N,T}) where {N, T<:RelaxTag} = x

zero(::Type{MCAffPnt{N,T}}) where {N,T} = MCAffPnt{N,T}(zero(MC{N,T}), zero(AffineEAGO{N}))
zero(x::MCAffPnt{N,T}) where {N,T} = MCAffPnt{N,T}(zero(x.v), zero(x.box))

one(::Type{MCAffPnt{N,T}}) where {N,T} = MCAffPnt{N,T}(one(MC{N,T}), one(AffineEAGO{N}))
one(x::MCAffPnt{N,T}) where {N,T} = MCAffPnt{N,T}(one(MC{N,T}), one(AffineEAGO{N}))

+(x::MCAffPnt{N,T}, y::MCAffPnt{N,T}) where {N,T} = MCAffPnt{N,T}(x.v + y.v, x.box + y.box)
+(x::MCAffPnt{N,T}, α::Number) where {N,T} = MCAffPnt{N,T}(x.v + α, x.box + α)
+(α::Number, x::MCAffPnt{N,T}) where {N,T} = x + α

*(x::MCAffPnt{N,T}, y::MCAffPnt{N,T}) where {N,T} = MCAffPnt{N,T}(x.v*y.v, x.box*y.box)
*(x::MCAffPnt{N,T}, α::Number) where {N,T} = MCAffPnt{N,T}(x.v * α, x.box * α)
*(α::Number, x::MCAffPnt{N,T}) where {N,T} = x*α

-(x::MCAffPnt{N,T}, y::MCAffPnt{N,T}) where {N,T} = MCAffPnt{N,T}(x.v-y.v, x.box-y.box)
-(x::MCAffPnt{N,T}) where {N,T} = MCAffPnt{N,T}(-x.v, -x.box)
-(x::MCAffPnt{N,T}, α::Number) where {N,T} = x + (-α)
-(α::Number, x::MCAffPnt{N,T}) where {N,T} = α + (-x)

/(x::MCAffPnt{N,T}, y::MCAffPnt{N,T}) where {N,T} = x*inv(y)
/(x::MCAffPnt{N,T}, α::T) where {N,T} = MCAffPnt{N,T}(x.v/α, x.box/α)
/(α::T, x::MCAffPnt{N,T}) where {N,T} = MCAffPnt{N,T}(α*inv(x.v), α*inv(x.box))

^(x::MCAffPnt{N,T}, n::Integer) where {N,T} = MCAffPnt{N,T}(x.v^n, x.box^n)
^(x::MCAffPnt{N,T}, n::Number) where {N,T} = MCAffPnt{N,T}(x.v^n, x.box^n)
inv(x::MCAffPnt{N,T}) where {N,T} = MCAffPnt{N,T}(inv(x.v), inv(x.box))

log(x::MCAffPnt{N,T}) where {N,T} = MCAffPnt{N,T}(log(x.v), log(x.box))
log10(x::MCAffPnt{N,T}) where {N,T} = MCAffPnt{N,T}(log10(x.v), log10(x.box))

exp(x::MCAffPnt{N,T}) where {N,T} = MCAffPnt{N,T}(exp(x.v), exp(x.box))
exp10(x::MCAffPnt{N,T}) where {N,T} = MCAffPnt{N,T}(exp10(x.v), exp10(x.box))


function extract_apriori_info(t::Union{RelaxAA,RelaxAAInfo}, v::VariableValues{Float64}, x::AffineEAGO{N}) where {N,T}
    z = 0.0
    for (k,i) in enumerate(t.v)
        l = _bd(v, i)
        u = ubd(v, i)
        z += x.γ[k]*(_val(v, i) - 0.5*(l + u))/(u - l)
    end
    Z = x.c + sum(z -> z*UNIT_INTERVAL, x.γ)
    xcv  = x.c + z - x.Δ
    xcc  = x.c + z + x.Δ
    xcvU = Z.hi - x.Δ
    xccL = Z.lo + x.Δ
    xcvg = x.γ
    xccg = x.γ
    return xcv, xcvU, xcc, xccL, xcvg, xccg
end

function _cut_info(v::VariableValues{Float64}, z::MC{N,T}, x::MCAffPnt{N,T}) where {N,T} 
    xcv, xcvU, xcc, xccL, xcvg, xccg = extract_apriori_info(RelaxAA(), v, x.box)
    zaff = MC{N,T}(xcv, xcc, Interval(x.box), xcvg, xccg, false)
    return zaff ∩ z
end
function _cut_info(v::VariableValues{Float64}, z::MCAffPnt{N,T}, x::MCAffPnt{N,T}) where {N,T} 
    xcv, xcvU, xcc, xccL, xcvg, xccg = extract_apriori_info(RelaxAA(), v, x.box)
    zaff = MC{N,T}(xcv, xcc, Interval(x.box), xcvg, xccg, false)
    return zaff ∩ z.v
end

function _cut(t::RelaxAAInfo, b, k, x::MCAffPnt{N,T}, z::MCAffPnt{N,T}, v::VariableValues, ϵ::Float64, s::Vector{Int}, c::Bool, p::Bool) where {N,T<:RelaxTag}
    xt = p ? MCAffPnt{N,T}(set_value_post(x.v, v, s, ϵ), x.box) : x
    xtmc = _cut_info(v, xt, xt)
    xtaffp = MCAffPnt{N,T}(xtmc, xt.box)
    b._info[k] = xtaffp
    return
end

function _cut(t::RelaxAA, b, k, x::MC{N,T}, z::MCAffPnt{N,T}, v::VariableValues, ϵ::Float64, s::Vector{Int}, c::Bool, p::Bool) where {N,T<:RelaxTag}
    xMC = set_value_post(c ? x ∩ x.Intv ∩ Interval(z.box) : x, v, s, ϵ)
    xt = p ? xMC : x
    zt = _cut_info(v, xt, info(b, k))
    _store_set!(b, zt, k)
    return
end

relax_info(s::RelaxAA, n::Int, t::T) where T = MCAffPnt{n,T}
function f_init!(t::RelaxAA, g::DAT, b::RelaxCache)
    tinfo = RelaxAAInfo(t.v)
    for k = node_count(g):-1:1
        c = node_class(g, k)
        if c == EXPRESSION
            fprop!(tinfo, Expression(), g, b, k)
        elseif c == VARIABLE
            fprop!(tinfo, Variable(), g, b, k)
        end
        b._set[k] = info(b, k).v
    end
end


function fprop!(t::RelaxAAInfo, vt::Variable, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    i = first_index(g, k)
    x = val(b, i)
    l = lbd(b, i)
    u = ubd(b, i)
    z = varset(MC{N,T}, rev_sparsity(g, i, k), x, x, l, u)
    zaff = AffineEAGO{N}(x, Interval(l,u), i)
    zinfo = MCAffPnt{N,T}(z, zaff)
    b._info[k] = zinfo
    return 
end
function fprop!(t::RelaxAA, vt::Variable, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    i = first_index(g, k)
    x = val(b, i)
    l = lbd(b, i)
    u = ubd(b, i)
    z = varset(MC{N,T}, rev_sparsity(g, i, k), x, x, l, u)
    z = z ∩ _interval(b, k)
    _store_set!(b, z, k)
end

info_or_set(t::RelaxAAInfo, b::RelaxCache{V,N,T}, k) where {V,N,T<:RelaxTag} = info(b, k)
info_or_set(t::RelaxAA, b::RelaxCache{V,N,T}, k) where {V,N,T<:RelaxTag} = set(b, k)
_store_out!(t::RelaxAAInfo, b::RelaxCache{V,N,T}, z, k) where {V,N,T<:RelaxTag} = (b._info[k] = z; nothing)
_store_out!(t::RelaxAA, b::RelaxCache{V,N,T}, z, k) where {V,N,T<:RelaxTag} = _store_set!(b, z, k)

function fprop!(t::Union{RelaxAAInfo,RelaxAA}, v::Val{EXP}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    x = info_or_set(t, b, child(g, 1, k))
    _cut(t, b, k, exp(x), info(b, k), b.v, zero(Float64), sparsity(g,k), b.cut, b.post)
end
function fprop!(t::Union{RelaxAAInfo,RelaxAA}, v::Val{EXP10}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    x = info_or_set(t, b, child(g, 1, k))
    _cut(t, b, k, exp10(x), info(b, k), b.ic.v, zero(Float64), sparsity(g,k), b.cut, b.post)
end
function fprop!(t::Union{RelaxAAInfo,RelaxAA}, v::Val{LOG}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    x = info_or_set(t, b, child(g, 1, k))
    _cut(t, b, k, log(x), info(b, k), b.ic.v, zero(Float64), sparsity(g,k), b.cut, b.post)
end

function fprop!(t::Union{RelaxAA,RelaxAAInfo}, v::Val{POW}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    x = child(g, 1, k)
    y = child(g, 2, k)
    x_is_num = is_num(b, x)
    y_is_num = is_num(b, y)
    if y_is_num && isone(num(b, y))
        z = info_or_set(t, b, x)
        store_out!(t, b, z, k)
    elseif y_is_num && iszero(num(b, y))
        store_out!(t, b, zero(info_or_set(t, b, x)), k)
    else
        if !x_is_num && y_is_num
            z = info_or_set(t, b, x)^num(b, y)
        elseif x_is_num && !y_is_num
            z = num(b, x)^info_or_set(t, b, y)
        elseif !x_is_num && !y_is_num
            z = info_or_set(t, b, x)^info_or_set(t, b, y)
        end
    _cut(t, b, k, z, info(b,k), b.ic.v, zero(Float64), sparsity(g,k), b.cut, b.post)
    end
end

function fprop!(t::Union{RelaxAA,RelaxAAInfo}, v::Val{MINUS}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    x = child(g, 1, k)
    x_is_num = is_num(b, x)
    if arity(g, k) == 2
        y = child(g, 2, k)
        y_is_num = is_num(b, y)
        if !x_is_num && y_is_num
            z = info_or_set(t, b, x) - num(b, y)
        elseif x_is_num && !y_is_num
            z = num(b, x) - info_or_set(t, b, y)
        else
            z = info_or_set(t, b, x) - info_or_set(t, b, y)
        end
    else
        z = -info_or_set(t, b, x)
    end
    _cut(t, b, k, z, info(b,k), b.ic.v, b.ϵ_sg, sparsity(g, k), b.cut, false)
end

function fprop!(t::Union{RelaxAA,RelaxAAInfo}, v::Val{DIV}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    x = child(g, 1, k)
    y = child(g, 2, k)
    if !is_num(b, x) && is_num(b, y)
        z = info_or_set(t, b, x)/num(b, y)
    elseif is_num(b, x) && !is_num(b, y)
        z = num(b, x)/info_or_set(t, b, y)
    else
        z = info_or_set(t, b, x)/info_or_set(t, b, y)
    end
    _cut(t, b, k, z, info(b,k), b.ic.v, b.ϵ_sg, sparsity(g, k), b.cut, false)
end

function fprop_2!(t::RelaxAAInfo, v::Val{PLUS}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    x = child(g, 1, k)
    y = child(g, 2, k)
    x_is_num = is_num(b, x)
    y_is_num = is_num(b, y)
    if !x_is_num && y_is_num
        z = info(b, x) + num(b, y)
    elseif x_is_num && !y_is_num
        z = num(b, x) + info(b, y)
    else
        z = info(b, x) + info(b, y)
    end
    _cut(t, b, k, z, info(b,k), b.ic.v, b.ϵ_sg, sparsity(g, k), b.cut, false)
end

function fprop_2!(t::RelaxAA, v::Val{PLUS}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    x = child(g, 1, k)
    y = child(g, 2, k)
    x_is_num = is_num(b, x)
    y_is_num = is_num(b, y)
    if !x_is_num && y_is_num
        z = set(b, x) + num(b, y)
    elseif x_is_num && !y_is_num
        z = num(b, x) + set(b, y)
    else
        z = set(b, x) + set(b, y)
    end
    _cut(t, b, k, z, info(b,k), b.ic.v, b.ϵ_sg, sparsity(g, k), b.cut, false)
end

function fprop_n!(t::RelaxAAInfo, ::Val{PLUS}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    z = one(V)
    znum = one(Float64)
    count = 0
    for i in children(g, k)
        if is_num(b, i)
            znum = znum*num(b, i)
        else
            z = z*info(b, i)
        end
        count += 1
    end
    z = z*znum
    _cut(t, b, k, z, info(b,k), b.ic.v, b.ϵ_sg, sparsity(g, k), b.cut, false)
end

function fprop_n!(t::RelaxAA, ::Val{PLUS}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    z = one(MC{N,T})
    znum = one(Float64)
    count = 0
    for i in children(g, k)
        if is_num(b, i)
            znum = znum*num(b, i)
        else
            z = z*set(t, b, i)
        end
        count += 1
    end
    z = z*znum
    _cut(t, b, k, z, info(b,k), b.ic.v, b.ϵ_sg, sparsity(g, k), b.cut, false)
end

function fprop_2!(t::RelaxAAInfo, v::Val{MULT}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    x = child(g, 1, k)
    y = child(g, 2, k)
    x_is_num = is_num(b, x)
    y_is_num = is_num(b, y)
    if !x_is_num && y_is_num
        zaff = *(info(b, x), num(b, y))
    elseif x_is_num && !y_is_num
        zaff = *(num(b, x), info(b, y))
    else
        xv = info(b, x)
        yv = info(b, y)
        xv_box = xv.box
        yv_box = yv.box
        z_box = xv_box*yv_box
        xcv, xcvU, xcc, xccL, xcvg, xccg = extract_apriori_info(RelaxAA(), b.v, xv_box)
        ycv, ycvU, ycc, yccL, ycvg, yccg = extract_apriori_info(RelaxAA(), b.v, yv_box)
        wIntv = xv.v.Intv*yv.v.Intv
        za_l = McCormick.mult_apriori_kernel(xv.v, yv.v, wIntv, xcv, ycv, xcvU, ycvU, xcvg, ycvg)
        za_u = McCormick.mult_apriori_kernel(-xv.v, -yv.v, wIntv, -xcc, -ycc, -xccL, -yccL, -xccg, -yccg)
        z = xv.v*yv.v 
        zaff = MCAffPnt{N,T}(z, z_box)                                        
    end
    _cut(t, b, k, zaff, info(b, k), b.ic.v, b.ϵ_sg, sparsity(g, k), b.cut, false)
end

function fprop_2!(t::RelaxAA, v::Val{MULT}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    x = child(g, 1, k)
    y = child(g, 2, k)
    x_is_num = is_num(b, x)
    y_is_num = is_num(b, y)
    if !x_is_num && y_is_num
        z = *(set(b, x), num(b, y))
    elseif x_is_num && !y_is_num
        z = *(num(b, x), set(b, y))
    else
        xs = _set(b, x)
        ys = _set(b, y)
        xinfo = info(b, x)
        yinfo = info(b, y)
        xv = xinfo
        yv = yinfo
        xcv, xcvU, xcc, xccL, xcvg, xccg = extract_apriori_info(RelaxAA(), b.v, xv.box)
        ycv, ycvU, ycc, yccL, ycvg, yccg = extract_apriori_info(RelaxAA(), b.v, yv.box)
        wIntv = xv.v.Intv*yv.v.Intv 
        za_l = McCormick.mult_apriori_kernel(xs, ys, wIntv, xcv, ycv, xcvU, ycvU, xcvg, ycvg)
        za_u = McCormick.mult_apriori_kernel(-xs, -ys, wIntv, -xcc, -ycc, -xccL, -yccL, -xccg, -yccg)
        z = (xs*ys) ∩ za_l ∩ za_u                                        
    end
    _cut(t, b, k, z, info(b,k), b.ic.v, b.ϵ_sg, sparsity(g, k), b.cut, false)
end

#TODO:
function fprop_n!(t::RelaxAAInfo, ::Val{MULT}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    z = one(V)
    znum = one(Float64)
    count = 0
    for i in children(g, k)
        if is_num(b, i)
            znum = znum*num(b, i)
        else
            z = z*info(b, i)
        end
        count += 1
    end
    z = z*znum
    _cut(t, b, k, z, info(b,k), b.ic.v, b.ϵ_sg, sparsity(g, k), b.cut, false)
end

function fprop_n!(t::RelaxAA, ::Val{MULT}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    ys = one(MC{N,T})
    znum = one(Float64)
    b._mult_temp = one(MCAffPnt{N,T})
    count = 0
    first_set = true
    for i in children(g, k)
        if is_num(b, i)
            znum = znum*num(b, i)
        else
            xs = _set(b, i)
            xinfo = info(b, i)
            xi = xinfo.v
            if !first_set
                ys = b._mult_temp.v
                xcvU, xccL = extract_apriori_info(t, b.v, xinfo.box)
                ycvU, yccL = extract_apriori_info(t, b.v, b._mult_temp.box)
                xcv = xi.cv;       ycv = b._mult_temp.v.cv
                xcc = xi.cc;       ycc = b._mult_temp.v.cc
                xcvg = xi.cv_grad; ycvg = b._mult_temp.v.cv_grad
                xccg = xi.cc_grad; yccg = b._mult_temp.v.cc_grad
                wIntv = xs.Intv*ys.Intv ∩ Interval(xinfo.box*b._mult_temp.box)
                za_l = McCormick.mult_apriori_kernel(xs, ys, wIntv, xcv, ycv, xcvU, ycvU, xcvg, ycvg)
                za_u = McCormick.mult_apriori_kernel(-xs, -ys, wIntv, -xcc, -ycc, -xccL, -yccL, -xccg, -yccg)
                ys = (xs*ys) ∩ za_l ∩ za_u
                b._mult_temp = MCAffPnt{N,T}(ys, b._mult_temp.box*xinfo.box)
            else
                first_set = false
                b._mult_temp = MCAffPnt{N,T}(xi, xinfo.box)
                ys = xs
            end
        end
        count += 1
    end
    z = ys*znum
    _cut(t, b, k, z, info(b,k), b.ic.v, b.ϵ_sg, sparsity(g, k), b.cut, false)
end

function fprop!(t::Union{RelaxAA,RelaxAAInfo}, v::Val{PLUS}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    (arity(g, k) == 2) ? fprop_2!(t, Val(PLUS), g, b, k) : fprop_n!(t, Val(PLUS), g, b, k)
end
function fprop!(t::Union{RelaxAA,RelaxAAInfo}, v::Val{MULT}, g::DAT, b::RelaxCache{V,N,T}, k::Int) where {V,N,T<:RelaxTag}
    (arity(g, k) == 2) ? fprop_2!(t, Val(MULT), g, b, k) : fprop_n!(t, Val(MULT), g, b, k)
end