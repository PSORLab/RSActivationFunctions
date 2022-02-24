using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -1.9515287248442508 <= q <= 2.66866690087751)

                     add_NL_constraint(m, :(log(1 + exp(0.7910255593621245 + -0.9427161680575309*log(1 + exp(-0.9881665794300214 + 0.6330293110094045*$(x[1]) + 0.8786704770929474*$(x[2]) + 0.11417841288574326*$(x[3]))) + 0.08805479609820388*log(1 + exp(0.6512390960566696 + -0.0007920608466527312*$(x[1]) + -0.28084831326063275*$(x[2]) + 0.17707194589795927*$(x[3]))))) + log(1 + exp(-0.7742678604770994 + 0.3023856535046492*log(1 + exp(-0.9881665794300214 + 0.6330293110094045*$(x[1]) + 0.8786704770929474*$(x[2]) + 0.11417841288574326*$(x[3]))) + -0.5348059615310992*log(1 + exp(0.6512390960566696 + -0.0007920608466527312*$(x[1]) + -0.28084831326063275*$(x[2]) + 0.17707194589795927*$(x[3]))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    