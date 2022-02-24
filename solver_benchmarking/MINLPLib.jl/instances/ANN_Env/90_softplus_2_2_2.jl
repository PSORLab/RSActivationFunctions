using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -2.3024137037198598 <= q <= 2.756640266423038)

                     add_NL_constraint(m, :(softplus(0.8745011162287222 + -0.4747792679508138*softplus(0.20979439694353186 + 0.2523963619940135*$(x[1]) + 0.6039440929888729*$(x[2])) + 0.06822380173369114*softplus(-0.192203947089018 + 0.35618645260012016*$(x[1]) + -0.9380253880003773*$(x[2]))) + softplus(-0.5284568968024108 + 0.8782860084347535*softplus(0.20979439694353186 + 0.2523963619940135*$(x[1]) + 0.6039440929888729*$(x[2])) + 0.990986450920027*softplus(-0.192203947089018 + 0.35618645260012016*$(x[1]) + -0.9380253880003773*$(x[2]))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    