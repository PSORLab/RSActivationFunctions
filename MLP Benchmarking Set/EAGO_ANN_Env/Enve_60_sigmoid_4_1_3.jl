using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -4.31029635958425 <= q <= 8.595176100608434)

                     add_NL_constraint(m, :(sigmoid(0.6100759138891441 + 0.00044091822468272923*$(x[1]) + -0.7032445138963812*$(x[2]) + 0.5215206000784547*$(x[3]) + -0.495581890824651*$(x[4])) + sigmoid(0.9388996959527955 + 0.6074711199556857*$(x[1]) + -0.6873002153557355*$(x[2]) + -0.9017167528600103*$(x[3]) + 0.4182953066465589*$(x[4])) + sigmoid(0.5934642606701521 + -0.5668521480134294*$(x[1]) + 0.4326572912225566*$(x[2]) + 0.38170112315458793*$(x[3]) + 0.7359543498636074*$(x[4])) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    