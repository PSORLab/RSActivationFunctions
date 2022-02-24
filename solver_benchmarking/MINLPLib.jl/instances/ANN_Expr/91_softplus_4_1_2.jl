using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -3.97969642451919 <= q <= 6.896074133539049)

                     add_NL_constraint(m, :(log(1 + exp(0.518279586272822 + -0.9811103721569414*$(x[1]) + -0.6250390763797746*$(x[2]) + -0.8382325906616188*$(x[3]) + 0.6951086592760602*$(x[4]))) + log(1 + exp(0.9399092682371073 + -0.6199086749993752*$(x[1]) + -0.7631209251662368*$(x[2]) + -0.009236754359899013*$(x[3]) + -0.9061282260292134*$(x[4]))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    