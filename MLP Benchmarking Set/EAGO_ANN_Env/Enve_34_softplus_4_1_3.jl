using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -7.42169870698135 <= q <= 6.885138706762119)

                     add_NL_constraint(m, :(softplus(-0.0222320204245543 + -0.5505852551566339*$(x[1]) + 0.13404876312842262*$(x[2]) + 0.7045051352351352*$(x[3]) + -0.5648308583931185*$(x[4])) + softplus(0.0010079916914387255 + -0.5377163842711337*$(x[1]) + -0.4753939063174619*$(x[2]) + 0.12452766426971351*$(x[3]) + 0.6191428886839083*$(x[4])) + softplus(-0.24705597137650015 + 0.7094379084666391*$(x[1]) + -0.9026669001117731*$(x[2]) + 0.8921464241238768*$(x[3]) + -0.9384166187139171*$(x[4])) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    