using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -188.1259937231292 <= q <= 182.27130445175948)


                     @objective(m, Min, q)

                     return m

                    