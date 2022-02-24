using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -2.888482713819031 <= q <= 4.953638237128964)

                     add_NL_constraint(m, :(softplus(0.8288775426207136 + -0.6672424360859748*$(x[1]) + 0.028347984312300234*$(x[2]) + 0.11850756042164479*$(x[3])) + softplus(-0.7336296978706387 + 0.2644686964706797*$(x[1]) + 0.2908846438936661*$(x[2]) + -0.6031271331048136*$(x[3])) + softplus(0.9373299169048916 + -0.6697043956054385*$(x[1]) + 0.9604003039018942*$(x[2]) + 0.3183773216775858*$(x[3])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    