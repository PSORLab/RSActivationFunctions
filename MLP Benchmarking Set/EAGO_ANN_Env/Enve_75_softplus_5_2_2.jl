using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -4.82520216793879 <= q <= 5.904061559483962)

                     add_NL_constraint(m, :(softplus(-0.16993797733768545 + -0.724771868611668*softplus(-0.8698603499897453 + 0.3983219553980155*$(x[1]) + -0.4759995744989074*$(x[2]) + 0.2803886196485439*$(x[3]) + 0.8328387318699799*$(x[4]) + -0.7607026514532116*$(x[5])) + 0.38013251292096273*softplus(0.8898725368493707 + 0.48421729845310235*$(x[1]) + -0.1057653651876902*$(x[2]) + -0.9691744048393858*$(x[3]) + -0.616942089331904*$(x[4]) + 0.43061640270575285*$(x[5]))) + softplus(0.5270289892130493 + 0.3913029381958246*softplus(-0.8698603499897453 + 0.3983219553980155*$(x[1]) + -0.4759995744989074*$(x[2]) + 0.2803886196485439*$(x[3]) + 0.8328387318699799*$(x[4]) + -0.7607026514532116*$(x[5])) + -0.5011978477457024*softplus(0.8898725368493707 + 0.48421729845310235*$(x[1]) + -0.1057653651876902*$(x[2]) + -0.9691744048393858*$(x[3]) + -0.616942089331904*$(x[4]) + 0.43061640270575285*$(x[5]))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    