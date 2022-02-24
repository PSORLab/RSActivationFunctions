using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -4.572700962608518 <= q <= 2.439967378874661)

                     add_NL_constraint(m, :(1/(1 + exp(-(-0.9934015186054679 + 0.8375762785867722*$(x[1]) + 0.668300383494036*$(x[2])))) + 1/(1 + exp(-(0.08494880908446678 + -0.07388095057292743*$(x[1]) + -0.9863741327473892*$(x[2])))) + 1/(1 + exp(-(-0.1579140823459273 + 0.7305802491259836*$(x[1]) + -0.2096221762144812*$(x[2])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    