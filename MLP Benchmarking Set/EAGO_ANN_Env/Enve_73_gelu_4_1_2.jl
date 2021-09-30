using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -2.4021571479975186 <= q <= 4.071540778649951)

                     add_NL_constraint(m, :(gelu(0.7829662217447719 + 0.07180483135667215*$(x[1]) + -0.5044045548843052*$(x[2]) + 0.650982891482824*$(x[3]) + -0.1854163655880261*$(x[4])) + gelu(0.05172559358144424 + -0.05147489590290055*$(x[1]) + -0.5299991937141408*$(x[2]) + -0.5668366590563694*$(x[3]) + -0.6759295713384965*$(x[4])) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    