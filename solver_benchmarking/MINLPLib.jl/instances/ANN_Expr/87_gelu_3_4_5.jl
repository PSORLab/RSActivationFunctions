using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -253.46990028842683 <= q <= 245.8372609992101)


                     @objective(m, Min, q)

                     return m

                    