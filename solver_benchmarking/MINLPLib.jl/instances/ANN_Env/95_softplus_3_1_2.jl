using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -2.9521310086708863 <= q <= 4.804083818674538)

                     add_NL_constraint(m, :(softplus(0.9022086576378046 + 0.4902152360110663*$(x[1]) + 0.6884954091678397*$(x[2]) + 0.4475675674383903*$(x[3])) + softplus(0.023767747364021208 + -0.6986395659548528*$(x[1]) + 0.5778907575747945*$(x[2]) + 0.9752988775257685*$(x[3])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    