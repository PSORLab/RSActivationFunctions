using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -282.6511122038421 <= q <= 285.8163858220877)


                     @objective(m, Min, q)

                     return m

                    