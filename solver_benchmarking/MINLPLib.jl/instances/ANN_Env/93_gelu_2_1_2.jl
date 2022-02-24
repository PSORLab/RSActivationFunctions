using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -2.8031108847289734 <= q <= 2.6473335630382056)

                     add_NL_constraint(m, :(gelu(-0.8475658981369358 + 0.9991334286088005*$(x[1]) + 0.6614487188155533*$(x[2])) + gelu(0.7696772372915519 + 0.2658705903136487*$(x[1]) + 0.798769486145587*$(x[2])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    