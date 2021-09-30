using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -1.4534286027830623 <= q <= 2.302458963782507)

                     add_NL_constraint(m, :(softplus(0.06808948413472526 + -0.4148124463386935*$(x[1]) + -0.6976597436360379*$(x[2])) + softplus(0.356425696364997 + 0.660419599091143*$(x[1]) + -0.10505199421691014*$(x[2])) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    