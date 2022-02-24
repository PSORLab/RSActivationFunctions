using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -12.266285154621722 <= q <= 12.793564807427844)

                     add_NL_constraint(m, :(sigmoid(-0.46956444239618067 + 0.18451073354487368*$(x[1]) + -0.3056648113863898*$(x[2]) + -0.8733496775879632*$(x[3]) + -0.7482422964857776*$(x[4]) + 0.39599386992219143*$(x[5])) + sigmoid(0.2831226968799312 + -0.000968797402465249*$(x[1]) + -0.6607077518367617*$(x[2]) + 0.11528579207894962*$(x[3]) + -0.7069435392252328*$(x[4]) + 0.7900850540395843*$(x[5])) + sigmoid(0.41681338081805386 + 0.17922182588365843*$(x[1]) + 0.5927695896331957*$(x[2]) + -0.635120363897093*$(x[3]) + 0.31189019033409826*$(x[4]) + 0.11999673266347255*$(x[5])) + sigmoid(-0.4140404108339504 + 0.583322476422016*$(x[1]) + 0.9921442381523282*$(x[2]) + 0.45546141054937284*$(x[3]) + -0.3096177212085385*$(x[4]) + 0.6860556002335301*$(x[5])) + sigmoid(0.44730860193520705 + -0.9763312564707065*$(x[1]) + 0.297857633230612*$(x[2]) + -0.7735590056574844*$(x[3]) + -0.3545666189247747*$(x[4]) + -0.48025799425370996*$(x[5])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    