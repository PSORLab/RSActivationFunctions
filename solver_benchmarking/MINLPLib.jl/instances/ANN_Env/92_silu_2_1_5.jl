using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -6.355742389270071 <= q <= 5.291892702058364)

                     add_NL_constraint(m, :(swish(-0.046987465146302476 + 0.9722813338228051*$(x[1]) + -0.7814511744189074*$(x[2])) + swish(-0.9625414337890552 + 0.5872522747234861*$(x[1]) + 0.38591887418496684*$(x[2])) + swish(0.22754410958881088 + -0.15392955055702906*$(x[1]) + -0.6389708691593214*$(x[2])) + swish(-0.10351168983583303 + 0.8921688569394703*$(x[1]) + -0.5909582391668393*$(x[2])) + swish(0.35357163557652704 + -0.6460660745152582*$(x[1]) + 0.17482029817613265*$(x[2])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    