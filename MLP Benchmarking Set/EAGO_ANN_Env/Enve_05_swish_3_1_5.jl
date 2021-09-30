using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -6.869646164993465 <= q <= 7.0714655252186285)

                     add_NL_constraint(m, :(swish(-0.6450548671784859 + -0.14658028327751849*$(x[1]) + 0.21973723374270415*$(x[2]) + -0.10462603596206099*$(x[3])) + swish(0.9988060519387356 + 0.5483831456269517*$(x[1]) + -0.4576570817619925*$(x[2]) + -0.802322422013074*$(x[3])) + swish(-0.8881651479189867 + 0.6223710744141155*$(x[1]) + -0.20251785105419406*$(x[2]) + -0.8445741608934845*$(x[3])) + swish(0.1500865500343016 + -0.05832873601464783*$(x[1]) + 0.5552072702537312*$(x[2]) + -0.7275761391800151*$(x[3])) + swish(0.4852370932370178 + -0.4990586469066609*$(x[1]) + -0.6077484785923084*$(x[2]) + 0.5738672854125868*$(x[3])) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    