using JuMP, EAGO

                     m = Model()

                     register(m, :tgelu, 1, tgelu, autodiff = true)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -3.780280297263572 <= q <= 5.748799400081819)

                     add_NL_constraint(m, :(tgelu(0.21079675738386427 + -0.2705743557902234*tgelu(-0.20895974491439562 + -0.15797187365092702*$(x[1]) + -0.39437708866301824*$(x[2]) + 0.7332038302487396*$(x[3]) + 0.901939891599072*$(x[4]) + 0.8259645047751909*$(x[5])) + -0.44709961351743166*tgelu(-0.014781663795297995 + 0.7573494419808999*$(x[1]) + -0.6212048554540632*$(x[2]) + 0.3633400988716353*$(x[3]) + 0.44149107716961034*$(x[4]) + -0.18729537361481352*$(x[5]))) + tgelu(0.7334837428413463 + 0.18083040269571926*tgelu(-0.20895974491439562 + -0.15797187365092702*$(x[1]) + -0.39437708866301824*$(x[2]) + 0.7332038302487396*$(x[3]) + 0.901939891599072*$(x[4]) + 0.8259645047751909*$(x[5])) + -0.988880661208337*tgelu(-0.014781663795297995 + 0.7573494419808999*$(x[1]) + -0.6212048554540632*$(x[2]) + 0.3633400988716353*$(x[3]) + 0.44149107716961034*$(x[4]) + -0.18729537361481352*$(x[5]))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    