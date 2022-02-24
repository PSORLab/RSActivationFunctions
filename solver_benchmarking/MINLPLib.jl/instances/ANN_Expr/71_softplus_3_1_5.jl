using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -5.25849806212995 <= q <= 4.375154350776637)

                     add_NL_constraint(m, :(log(1 + exp(0.4440484214047524 + 0.27117166180904073*$(x[1]) + 0.1549579763005995*$(x[2]) + -0.19609804892644567*$(x[3]))) + log(1 + exp(-0.1025904891815026 + 0.10197317322059929*$(x[1]) + 0.049423996875141984*$(x[2]) + -0.5007584267765015*$(x[3]))) + log(1 + exp(0.34943625748753826 + 0.7627977483870185*$(x[1]) + -0.7008024309541692*$(x[2]) + -0.24608961291909015*$(x[3]))) + log(1 + exp(-0.3923242036950221 + -0.06896448884223272*$(x[1]) + -0.2210802938542722*$(x[2]) + -0.35764467125421584*$(x[3]))) + log(1 + exp(-0.7402418416924226 + -0.21635543628426968*$(x[1]) + -0.713523096277699*$(x[2]) + 0.2551851437719974*$(x[3]))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    