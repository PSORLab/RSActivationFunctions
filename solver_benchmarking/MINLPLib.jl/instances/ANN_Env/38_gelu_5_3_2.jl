using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -3.1544947018849654 <= q <= 4.066926790342683)

                     add_NL_constraint(m, :(gelu(-0.7544717811096784 + -0.6289249660430758*gelu(-0.401848603561894 + 0.9879850116297502*gelu(0.7471741672319268 + 0.1562161389212755*$(x[1]) + -0.06104000939937393*$(x[2]) + -0.24396039580062512*$(x[3]) + -0.35002310972719863*$(x[4]) + -0.47890036372503086*$(x[5])) + -0.6833595346541679*gelu(0.07659783311726898 + 0.1400377608602934*$(x[1]) + -0.5920144548543678*$(x[2]) + -0.3715997206073691*$(x[3]) + -0.08423829417739581*$(x[4]) + -0.48744433011047184*$(x[5]))) + -0.656030629474813*gelu(-0.9665555719654946 + 0.014645656858022438*gelu(0.7471741672319268 + 0.1562161389212755*$(x[1]) + -0.06104000939937393*$(x[2]) + -0.24396039580062512*$(x[3]) + -0.35002310972719863*$(x[4]) + -0.47890036372503086*$(x[5])) + -0.5583055948877655*gelu(0.07659783311726898 + 0.1400377608602934*$(x[1]) + -0.5920144548543678*$(x[2]) + -0.3715997206073691*$(x[3]) + -0.08423829417739581*$(x[4]) + -0.48744433011047184*$(x[5])))) + gelu(-0.3137145052115069 + -0.6627216812930383*gelu(-0.401848603561894 + 0.9879850116297502*gelu(0.7471741672319268 + 0.1562161389212755*$(x[1]) + -0.06104000939937393*$(x[2]) + -0.24396039580062512*$(x[3]) + -0.35002310972719863*$(x[4]) + -0.47890036372503086*$(x[5])) + -0.6833595346541679*gelu(0.07659783311726898 + 0.1400377608602934*$(x[1]) + -0.5920144548543678*$(x[2]) + -0.3715997206073691*$(x[3]) + -0.08423829417739581*$(x[4]) + -0.48744433011047184*$(x[5]))) + -0.39516873731093893*gelu(-0.9665555719654946 + 0.014645656858022438*gelu(0.7471741672319268 + 0.1562161389212755*$(x[1]) + -0.06104000939937393*$(x[2]) + -0.24396039580062512*$(x[3]) + -0.35002310972719863*$(x[4]) + -0.47890036372503086*$(x[5])) + -0.5583055948877655*gelu(0.07659783311726898 + 0.1400377608602934*$(x[1]) + -0.5920144548543678*$(x[2]) + -0.3715997206073691*$(x[3]) + -0.08423829417739581*$(x[4]) + -0.48744433011047184*$(x[5])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    