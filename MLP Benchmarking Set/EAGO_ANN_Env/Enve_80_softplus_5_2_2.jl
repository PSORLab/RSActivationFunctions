using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -3.20865271522045 <= q <= 3.160464133723557)

                     add_NL_constraint(m, :(softplus(-0.36514571395935347 + -0.8803367291517636*softplus(-0.7448341961978344 + 0.2543445682341523*$(x[1]) + 0.06967038565571171*$(x[2]) + -0.052522391704918014*$(x[3]) + -0.3645464271604002*$(x[4]) + -0.720478546123605*$(x[5])) + 0.6588350198197719*softplus(-0.20869206372593885 + -0.2771121043499276*$(x[1]) + -0.6488845295471224*$(x[2]) + 0.24689145124581335*$(x[3]) + -0.843095003344259*$(x[4]) + 0.23074288956911015*$(x[5]))) + softplus(-0.3485741278815606 + -0.23871421734816023*softplus(-0.7448341961978344 + 0.2543445682341523*$(x[1]) + 0.06967038565571171*$(x[2]) + -0.052522391704918014*$(x[3]) + -0.3645464271604002*$(x[4]) + -0.720478546123605*$(x[5])) + 0.030610752958533105*softplus(-0.20869206372593885 + -0.2771121043499276*$(x[1]) + -0.6488845295471224*$(x[2]) + 0.24689145124581335*$(x[3]) + -0.843095003344259*$(x[4]) + 0.23074288956911015*$(x[5]))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    