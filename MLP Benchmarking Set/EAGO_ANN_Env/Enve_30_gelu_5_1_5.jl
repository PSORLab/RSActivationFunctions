using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -13.481388893179142 <= q <= 12.014219536498445)

                     add_NL_constraint(m, :(gelu(-0.5737511174515264 + -0.10438576569225866*$(x[1]) + 0.9913707311557634*$(x[2]) + 0.9611980749955134*$(x[3]) + 0.3639646143364126*$(x[4]) + -0.759150925526566*$(x[5])) + gelu(0.17042194148200895 + -0.30600173191140456*$(x[1]) + -0.18735526509721678*$(x[2]) + 0.06707011813513697*$(x[3]) + 0.2412814807501391*$(x[4]) + 0.5279269217974258*$(x[5])) + gelu(-0.09848982900952752 + -0.32683914565343386*$(x[1]) + 0.6648949663150878*$(x[2]) + 0.06576189076515426*$(x[3]) + -0.12621741856002622*$(x[4]) + -0.3743379137905096*$(x[5])) + gelu(-0.6286144700011276 + -0.9670305943186137*$(x[1]) + -0.39099126153248287*$(x[2]) + 0.555542735990032*$(x[3]) + 0.8251520619398258*$(x[4]) + 0.7056656077317189*$(x[5])) + gelu(0.39684879663982375 + 0.9115046103032514*$(x[1]) + -0.8945255069165534*$(x[2]) + -0.2760812020926*$(x[3]) + 0.3089499319240421*$(x[4]) + 0.8446037376076232*$(x[5])) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    