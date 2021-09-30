using JuMP, EAGO

                     m = Model()

                     register(m, :tgelu, 1, tgelu, autodiff = true)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -6.939257181647966 <= q <= 5.573028250184293)

                     add_NL_constraint(m, :(tgelu(-0.6054360344413348 + 0.7036671137163975*$(x[1]) + 0.6684135303762226*$(x[2]) + 0.7563055665714926*$(x[3])) + tgelu(0.206091372065635 + 0.24207270665612723*$(x[1]) + 0.9994655952247489*$(x[2]) + 0.8612264139937795*$(x[3])) + tgelu(-0.28376980335613666 + -0.9343368963416219*$(x[1]) + -0.16345538863750608*$(x[2]) + 0.9271995043982324*$(x[3])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    