using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -6.563131467197746 <= q <= 9.310824610097802)

                     add_NL_constraint(m, :(1/(1 + exp(-(0.5323461867533106 + 0.6233251355511138*1/(1 + exp(-(0.9117705078314118 + -0.4325489455874667*$(x[1]) + 0.44461795780922087*$(x[2]) + -0.310674968881143*$(x[3]) + 0.3874148238299093*$(x[4])))) + 0.13135733952437967*1/(1 + exp(-(0.4793544889478012 + 0.21125743039643652*$(x[1]) + -0.625324840825729*$(x[2]) + -0.9496988785997882*$(x[3]) + 0.8662818947925528*$(x[4])))) + -0.0038770043009219712*1/(1 + exp(-(0.7417071748063861 + 0.06375566706279434*$(x[1]) + -0.03899151888275032*$(x[2]) + 0.7226457302550271*$(x[3]) + 0.006005128865761655*$(x[4]))))))) + 1/(1 + exp(-(0.48577545639109676 + -0.6684827900810446*1/(1 + exp(-(0.9117705078314118 + -0.4325489455874667*$(x[1]) + 0.44461795780922087*$(x[2]) + -0.310674968881143*$(x[3]) + 0.3874148238299093*$(x[4])))) + 0.9262921195438665*1/(1 + exp(-(0.4793544889478012 + 0.21125743039643652*$(x[1]) + -0.625324840825729*$(x[2]) + -0.9496988785997882*$(x[3]) + 0.8662818947925528*$(x[4])))) + 0.29789654203123295*1/(1 + exp(-(0.7417071748063861 + 0.06375566706279434*$(x[1]) + -0.03899151888275032*$(x[2]) + 0.7226457302550271*$(x[3]) + 0.006005128865761655*$(x[4]))))))) + 1/(1 + exp(-(0.2863540742777313 + -0.41164742912319063*1/(1 + exp(-(0.9117705078314118 + -0.4325489455874667*$(x[1]) + 0.44461795780922087*$(x[2]) + -0.310674968881143*$(x[3]) + 0.3874148238299093*$(x[4])))) + 0.6047572744817185*1/(1 + exp(-(0.4793544889478012 + 0.21125743039643652*$(x[1]) + -0.625324840825729*$(x[2]) + -0.9496988785997882*$(x[3]) + 0.8662818947925528*$(x[4])))) + -0.7133357903136108*1/(1 + exp(-(0.7417071748063861 + 0.06375566706279434*$(x[1]) + -0.03899151888275032*$(x[2]) + 0.7226457302550271*$(x[3]) + 0.006005128865761655*$(x[4]))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    