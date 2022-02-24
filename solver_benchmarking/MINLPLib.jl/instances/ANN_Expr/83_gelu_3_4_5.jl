using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -70.32410580340233 <= q <= 71.1174998159228)


                     @objective(m, Min, q)

                     return m

                    