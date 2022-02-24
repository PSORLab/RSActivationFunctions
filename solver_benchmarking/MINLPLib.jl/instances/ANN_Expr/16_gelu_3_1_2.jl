using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -3.9360745052223884 <= q <= 4.0522696599704755)

                     add_NL_constraint(m, :((-0.02759545635089289 + 0.8396153655678034*$(x[1]) + -0.45360175821557247*$(x[2]) + 0.6295131713145015*$(x[3]))*(1 + erf((-0.02759545635089289 + 0.8396153655678034*$(x[1]) + -0.45360175821557247*$(x[2]) + 0.6295131713145015*$(x[3]))/sqrt(2)))/2 + (0.0856930337249362 + -0.9280577687232072*$(x[1]) + 0.795858881094551*$(x[2]) + -0.3475251376807962*$(x[3]))*(1 + erf((0.0856930337249362 + -0.9280577687232072*$(x[1]) + 0.795858881094551*$(x[2]) + -0.3475251376807962*$(x[3]))/sqrt(2)))/2 - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    