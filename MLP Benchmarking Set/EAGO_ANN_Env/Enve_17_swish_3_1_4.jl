using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -5.190142618859154 <= q <= 8.620713416828346)

                     add_NL_constraint(m, :(swish(-0.2007635572505131 + -0.6607260649489035*$(x[1]) + -0.8915281314843191*$(x[2]) + 0.8835454835456908*$(x[3])) + swish(0.3305374699269539 + 0.905231994954971*$(x[1]) + 0.633460614432312*$(x[2]) + -0.22756907090033351*$(x[3])) + swish(0.9514024554937319 + 0.5894540253931897*$(x[1]) + 0.06397102949839084*$(x[2]) + -0.7972172479600044*$(x[3])) + swish(0.6341090308144235 + -0.3435208508276264*$(x[1]) + 0.7247586575607596*$(x[2]) + -0.18444484633724834*$(x[3])) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    