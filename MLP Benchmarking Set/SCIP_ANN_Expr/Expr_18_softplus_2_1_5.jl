using JuMP, EAGO

                     m = Model()

                     register(m, :tsoftplus, 1, tsoftplus, autodiff = true)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -5.450963384108388 <= q <= 4.652881270054792)

                     add_NL_constraint(m, :(tsoftplus(0.37515279458139483 + -0.3635903626731438*$(x[1]) + 0.9624155915055943*$(x[2])) + tsoftplus(-0.4515475140682432 + -0.4605983563949265*$(x[1]) + 0.7202394179160163*$(x[2])) + tsoftplus(0.14371369704563985 + 0.5607814398395994*$(x[1]) + -0.0011605274328836401*$(x[2])) + tsoftplus(-0.5424016803138607 + -0.7462290198213233*$(x[1]) + -0.5181350688662887*$(x[2])) + tsoftplus(0.07604164572827132 + -0.6725953959487132*$(x[1]) + 0.04617714668310047*$(x[2])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    