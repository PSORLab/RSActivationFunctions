using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -2.766903106924556 <= q <= 4.509204784150574)

                     add_NL_constraint(m, :(softplus(-0.014453613120027686 + -0.7509515817850838*$(x[1]) + -0.4057877137389023*$(x[2]) + 0.9991671824019659*$(x[3])) + softplus(0.8856044517330366 + 0.6066049884732889*$(x[1]) + -0.0647975113133823*$(x[2]) + -0.8107449678249417*$(x[3])) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    