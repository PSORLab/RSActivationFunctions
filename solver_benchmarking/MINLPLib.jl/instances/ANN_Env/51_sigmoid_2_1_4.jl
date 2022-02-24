using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -3.611664788975792 <= q <= 3.6444325166036777)

                     add_NL_constraint(m, :(sigmoid(-0.4383440965768486 + -0.2075365332338559*$(x[1]) + 0.46530061880514095*$(x[2])) + sigmoid(0.22314644246023452 + -0.6314968564820513*$(x[1]) + 0.5512122876813113*$(x[2])) + sigmoid(0.30053153463966176 + -0.15357242883325783*$(x[1]) + 0.254162988138138*$(x[2])) + sigmoid(-0.06895001670910483 + 0.7534492827978192*$(x[1]) + -0.6113176568181604*$(x[2])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    