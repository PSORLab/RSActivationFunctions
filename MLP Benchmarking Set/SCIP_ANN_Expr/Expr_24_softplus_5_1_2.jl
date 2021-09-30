using JuMP, EAGO

                     m = Model()

                     register(m, :tsoftplus, 1, tsoftplus, autodiff = true)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -4.916565602999575 <= q <= 6.9781134802860425)

                     add_NL_constraint(m, :(tsoftplus(0.8691713791247122 + 0.7405102841110174*$(x[1]) + 0.7256975201560469*$(x[2]) + 0.8993166755469719*$(x[3]) + 0.728816395435476*$(x[4]) + -0.8154484230897734*$(x[5])) + tsoftplus(0.1616025595185211 + -0.1154332197819059*$(x[1]) + 0.3222302519848994*$(x[2]) + 0.43761654119401605*$(x[3]) + -0.7771770959686237*$(x[4]) + 0.38509313437407755*$(x[5])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    