using JuMP, EAGO

                     m = Model()

                     register(m, :tsigmoid, 1, tsigmoid, autodiff = true)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -5.208761562020471 <= q <= 1.475335367709365)

                     add_NL_constraint(m, :(tsigmoid(-0.9840102842485683 + 0.4734141195729369*$(x[1]) + 0.6213348303145736*$(x[2]) + -0.4749354108648576*$(x[3])) + tsigmoid(-0.8827028129069845 + -0.28599354461480386*$(x[1]) + 0.997399616063869*$(x[2]) + -0.4889709434338769*$(x[3])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    