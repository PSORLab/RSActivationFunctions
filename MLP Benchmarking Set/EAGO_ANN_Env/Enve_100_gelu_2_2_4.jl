using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -8.105030757502986 <= q <= 10.49717988998953)

                     add_NL_constraint(m, :(gelu(0.5325666967121063 + 0.9569139064469576*gelu(0.7942776101229203 + 0.9757716251128614*$(x[1]) + -0.3937340841449166*$(x[2])) + 0.0539906086084283*gelu(0.20197462337253302 + -0.5315483329386361*$(x[1]) + 0.005578390753937956*$(x[2])) + -0.5455435621158489*gelu(-0.5110702462764887 + -0.8955201000882371*$(x[1]) + 0.28916475745118664*$(x[2])) + 0.10760605046273009*gelu(-0.2921837466860855 + -0.9093829251401608*$(x[1]) + -0.45868023406514213*$(x[2]))) + gelu(-0.8875633577473665 + 0.02440462199726401*gelu(0.7942776101229203 + 0.9757716251128614*$(x[1]) + -0.3937340841449166*$(x[2])) + -0.2402252876551083*gelu(0.20197462337253302 + -0.5315483329386361*$(x[1]) + 0.005578390753937956*$(x[2])) + -0.9618468400117499*gelu(-0.5110702462764887 + -0.8955201000882371*$(x[1]) + 0.28916475745118664*$(x[2])) + 0.435602530530228*gelu(-0.2921837466860855 + -0.9093829251401608*$(x[1]) + -0.45868023406514213*$(x[2]))) + gelu(-0.43605610553032603 + -0.5874975010156072*gelu(0.7942776101229203 + 0.9757716251128614*$(x[1]) + -0.3937340841449166*$(x[2])) + 0.9757732655302038*gelu(0.20197462337253302 + -0.5315483329386361*$(x[1]) + 0.005578390753937956*$(x[2])) + -0.9381894013502095*gelu(-0.5110702462764887 + -0.8955201000882371*$(x[1]) + 0.28916475745118664*$(x[2])) + -0.5437351637336407*gelu(-0.2921837466860855 + -0.9093829251401608*$(x[1]) + -0.45868023406514213*$(x[2]))) + gelu(-0.4285617553077463 + 0.8597885686062003*gelu(0.7942776101229203 + 0.9757716251128614*$(x[1]) + -0.3937340841449166*$(x[2])) + -0.28541293234894205*gelu(0.20197462337253302 + -0.5315483329386361*$(x[1]) + 0.005578390753937956*$(x[2])) + 0.12197686628678062*gelu(-0.5110702462764887 + -0.8955201000882371*$(x[1]) + 0.28916475745118664*$(x[2])) + -0.446555574111807*gelu(-0.2921837466860855 + -0.9093829251401608*$(x[1]) + -0.45868023406514213*$(x[2]))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    