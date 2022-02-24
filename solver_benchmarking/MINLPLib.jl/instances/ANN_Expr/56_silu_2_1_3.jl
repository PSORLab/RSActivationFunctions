using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -3.0943792528344187 <= q <= 2.261396294653902)

                     add_NL_constraint(m, :((-0.6000573802834048 + -0.4460703798694121*$(x[1]) + -0.8919582737388767*$(x[2]))/(1 + exp(-(-0.6000573802834048 + -0.4460703798694121*$(x[1]) + -0.8919582737388767*$(x[2])))) + (0.4850728208540551 + 0.8239401438043505*$(x[1]) + -0.080858577965925*$(x[2]))/(1 + exp(-(0.4850728208540551 + 0.8239401438043505*$(x[1]) + -0.080858577965925*$(x[2])))) + (-0.3015069196609086 + 0.14649274768979348*$(x[1]) + 0.2885676506758026*$(x[2]))/(1 + exp(-(-0.3015069196609086 + 0.14649274768979348*$(x[1]) + 0.2885676506758026*$(x[2])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    