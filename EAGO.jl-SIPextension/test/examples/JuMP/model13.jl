using JuMP, EAGO, Gurobi

m = Model(with_optimizer(EAGO.Optimizer, relaxed_optimizer = Gurobi.Optimizer(OutputFlag=0)))

# ----- Variables ----- #
x_Idx = Any[1, 2, 3, 4, 5, 6]
@variable(m, x[x_Idx])
JuMP.set_lower_bound(x[5], 0.0)
JuMP.set_lower_bound(x[1], 0.0)
JuMP.set_lower_bound(x[4], 0.0)
JuMP.set_lower_bound(x[2], 0.0)
JuMP.set_lower_bound(x[6], 0.0)
JuMP.set_lower_bound(x[3], 0.0)
JuMP.set_upper_bound(x[1], 1.0)
JuMP.set_upper_bound(x[2], 2.0)
JuMP.set_upper_bound(x[3], 1.0)
JuMP.set_upper_bound(x[4], 4.0)
JuMP.set_upper_bound(x[5], 2.0)
JuMP.set_upper_bound(x[6], 6.0)

# ----- Objective ----- #
@NLobjective(m, Min, ( (2.5134-x[1]-x[3]-x[5])^2+ (2.044333373291-(exp(-0.05*x[2])*x[1]+exp(-0.05*x[4])*x[3]+exp(-0.05*x[6])*x[5]))^2+ (1.668404436564-(exp(-0.1*x[2])*x[1]+exp(-0.1*x[4])*x[3]+exp(-0.1*x[6])*x[5]))^2+ (1.366418021208-(exp(-0.15*x[2])*x[1]+exp(-0.15*x[4])*x[3]+exp(-0.15*x[6])*x[5]))^2+ (1.123232487372-(exp(-0.2*x[2])*x[1]+exp(-0.2*x[4])*x[3]+exp(-0.2*x[6])*x[5]))^2+ (0.9268897180037-(exp(-0.25*x[2])*x[1]+exp(-0.25*x[4])*x[3]+exp(-0.25*x[6])*x[5]))^2+ (0.7679338563728-(exp(-0.3*x[2])*x[1]+exp(-0.3*x[4])*x[3]+exp(-0.3*x[6])*x[5]))^2+ (0.6388775523106-(exp(-0.35*x[2])*x[1]+exp(-0.35*x[4])*x[3]+exp(-0.35*x[6])*x[5]))^2+ (0.5337835317402-(exp(-0.4*x[2])*x[1]+exp(-0.4*x[4])*x[3]+exp(-0.4*x[6])*x[5]))^2+ (0.4479363617347-(exp(-0.45*x[2])*x[1]+exp(-0.45*x[4])*x[3]+exp(-0.45*x[6])*x[5]))^2+ (0.377584788435-(exp(-0.5*x[2])*x[1]+exp(-0.5*x[4])*x[3]+exp(-0.5*x[6])*x[5]))^2+ (0.3197393199326-(exp(-0.55*x[2])*x[1]+exp(-0.55*x[4])*x[3]+exp(-0.55*x[6])*x[5]))^2+ (0.2720130773746-(exp(-0.6*x[2])*x[1]+exp(-0.6*x[4])*x[3]+exp(-0.6*x[6])*x[5]))^2+ (0.2324965529032-(exp(-0.65*x[2])*x[1]+exp(-0.65*x[4])*x[3]+exp(-0.65*x[6])*x[5]))^2+ (0.1996589546065-(exp(-0.7*x[2])*x[1]+exp(-0.7*x[4])*x[3]+exp(-0.7*x[6])*x[5]))^2+ (0.1722704126914-(exp(-0.75*x[2])*x[1]+exp(-0.75*x[4])*x[3]+exp(-0.75*x[6])*x[5]))^2+ (0.1493405660168-(exp(-0.8*x[2])*x[1]+exp(-0.8*x[4])*x[3]+exp(-0.8*x[6])*x[5]))^2+ (0.1300700206922-(exp(-0.85*x[2])*x[1]+exp(-0.85*x[4])*x[3]+exp(-0.85*x[6])*x[5]))^2+ (0.1138119324644-(exp(-0.9*x[2])*x[1]+exp(-0.9*x[4])*x[3]+exp(-0.9*x[6])*x[5]))^2+ (0.1000415587559-(exp(-0.95*x[2])*x[1]+exp(-0.95*x[4])*x[3]+exp(-0.95*x[6])*x[5]))^2+ (0.0883320908454-(exp(-x[2])*x[1]+exp(-x[4])*x[3]+exp(-x[6])*x[5]))^2+ (0.0783354401935-(exp(-1.05*x[2])*x[1]+exp(-1.05*x[4])*x[3]+exp(-1.05*x[6])*x[5]))^2+ (0.06976693743449-(exp(-1.1*x[2])*x[1]+exp(-1.1*x[4])*x[3]+exp(-1.1*x[6])*x[5]))^2+ (0.06239312536719-(exp(-1.15*x[2])*x[1]+exp(-1.15*x[4])*x[3]+exp(-1.15*x[6])*x[5]))^2))

JuMP.optimize!(m)
run_time = backend(m).optimizer.model.optimizer._run_time
println("run time: $run_time")
