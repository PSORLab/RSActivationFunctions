using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -6.018373055247795 <= q <= 6.487803046241799)

                     add_NL_constraint(m, :(swish(0.19376851627428104 + 0.3933491073480768*$(x[1]) + 0.4793157854621084*$(x[2]) + -0.33838846007485035*$(x[3])) + swish(-0.15524054815926425 + -0.40567893969528823*$(x[1]) + 0.8712457795772197*$(x[2]) + 0.12992564460855016*$(x[3])) + swish(0.3161341032413989 + 0.3735638789018374*$(x[1]) + 0.10933316417608863*$(x[2]) + -0.7034260445138099*$(x[3])) + swish(-0.11994707585941367 + -0.7986459872602469*$(x[1]) + 0.9497495939341021*$(x[2]) + 0.7004656651926187*$(x[3])) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    