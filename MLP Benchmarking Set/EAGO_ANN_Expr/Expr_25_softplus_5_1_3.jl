using JuMP, EAGO

                     m = Model()

                     register(m, :tsoftplus, 1, tsoftplus, autodiff = true)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -7.502635754871525 <= q <= 7.0795639334426435)

                     add_NL_constraint(m, :(tsoftplus(0.5467235682411733 + -0.7877500195011833*$(x[1]) + 0.16651618869185913*$(x[2]) + 0.8542848532738923*$(x[3]) + -0.22091303016582842*$(x[4]) + -0.8889389196086315*$(x[5])) + tsoftplus(-0.8054927319123402 + 0.07424589103750989*$(x[1]) + -0.8037750248709736*$(x[2]) + 0.3767939810393637*$(x[3]) + -0.9814016211409027*$(x[4]) + -0.7569087691868952*$(x[5])) + tsoftplus(0.04723325295672609 + -0.16757573666924452*$(x[1]) + 0.4128100299687696*$(x[2]) + 0.06547306071324543*$(x[3]) + 0.24561420340505835*$(x[4]) + 0.48809851488372624*$(x[5])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    