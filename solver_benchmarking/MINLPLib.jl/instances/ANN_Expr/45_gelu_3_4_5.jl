using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -96.83657520547105 <= q <= 99.93991688919364)


                     @objective(m, Min, q)

                     return m

                    