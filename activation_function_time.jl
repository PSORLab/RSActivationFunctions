
using BenchmarkTools, McCormick, Random
using Statistics: std

# Note: Run time is still random so results are't strictly reproducable...
Random.seed!(1234)

ex_softplus(x) = log(1 + exp(x))
ex_maxsig(x) =  max(x, 1/(1 + exp(-(x))))
ex_maxtanh(x) = max(x,tanh(x))
ex_softsign(x) = (x)/(1 + x)
ex_sigmoid(x) = 1/(1 + exp(-(x)))
ex_silu(x) = (x)/(1 + exp(-(x)))
ex_gelu(x) = (x)*(1 + erf((x)/sqrt(2)))/2

const TRIAL_REPEAT = 10000

function loop(func, x)
    foreach(i -> func(x), 1:TRIAL_REPEAT)
end
function summarize!(n, f_trial, ex_trial)
    env_min = median(f_trial.times)/TRIAL_REPEAT
    expr_min = median(ex_trial.times)/TRIAL_REPEAT
    ratio_percent = 100*(env_min/expr_min - 1)
    println("$n | env = $(env_min) | exp = $(expr_min) | ratio = $(ratio_percent)")
end
#start_mc() = MC{1,NS}(0.0, Interval(-3,3), 1)
start_mc() = MC{1,NS}(6*rand()-3, Interval(-3,3), 1)

f_trial = @benchmark loop(softplus,x) setup=(x=start_mc())
ex_trial = @benchmark loop(ex_softplus,x) setup=(x=start_mc())
summarize!("Softplus", f_trial, ex_trial)

f_trial = @benchmark loop(maxsig,x) setup=(x=start_mc())
ex_trial = @benchmark loop(ex_maxsig,x) setup=(x=start_mc())
summarize!("Maxsig", f_trial, ex_trial)

f_trial = @benchmark loop(maxtanh,x) setup=(x=start_mc())
ex_trial = @benchmark loop(ex_maxtanh,x) setup=(x=start_mc())
summarize!("MaxTanh", f_trial, ex_trial)

f_trial = @benchmark loop(softsign,x) setup=(x=start_mc())
ex_trial = @benchmark loop(ex_softsign,x) setup=(x=start_mc())
summarize!("Softsign", f_trial, ex_trial)

f_trial = @benchmark loop(sigmoid,x) setup=(x=start_mc())
ex_trial = @benchmark loop(ex_sigmoid,x) setup=(x=start_mc())
summarize!("Sigmoid", f_trial, ex_trial)

f_trial = @benchmark loop(swish,x) setup=(x=start_mc())
ex_trial = @benchmark loop(ex_silu,x) setup=(x=start_mc())
summarize!("Silu", f_trial, ex_trial)

f_trial = @benchmark loop(gelu,x) setup=(x=start_mc())
ex_trial = @benchmark loop(ex_gelu,x) setup=(x=start_mc())
summarize!("Gelu", f_trial, ex_trial)