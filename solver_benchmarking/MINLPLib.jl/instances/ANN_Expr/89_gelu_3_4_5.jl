using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -87.82140256746152 <= q <= 82.66761573703552)


                     @objective(m, Min, q)

                     return m

                    