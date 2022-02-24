using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -9.06497171739458 <= q <= 7.221276301954956)

                     add_NL_constraint(m, :(1/(1 + exp(-(0.6131370972312755 + 0.7644530725075485*$(x[1]) + 0.7481034782956382*$(x[2]) + 0.43160377180111187*$(x[3])))) + 1/(1 + exp(-(-0.2235485813631728 + -0.4821458309603419*$(x[1]) + -0.0034703029219445014*$(x[2]) + 0.5577106220080164*$(x[3])))) + 1/(1 + exp(-(0.3888102587500133 + 0.8414638325842052*$(x[1]) + 0.671097918031764*$(x[2]) + -0.05238562929678103*$(x[3])))) + 1/(1 + exp(-(-0.9543429273497122 + 0.7406334788256062*$(x[1]) + 0.9764525500434011*$(x[2]) + 0.3990593176587991*$(x[3])))) + 1/(1 + exp(-(-0.7459035549882156 + 0.4045099462664137*$(x[1]) + 0.9504581535443215*$(x[2]) + -0.1195761049288735*$(x[3])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    