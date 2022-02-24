using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -3.412981244406108 <= q <= 4.209108534415106)

                     add_NL_constraint(m, :(softplus(-0.4372399910599052 + -0.43792093299500046*$(x[1]) + -0.746120027144912*$(x[2]) + 0.8963217772222452*$(x[3])) + softplus(0.8353036360644039 + 0.6875424158053405*$(x[1]) + 0.5184113900400429*$(x[2]) + -0.5247283462030659*$(x[3])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    