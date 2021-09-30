using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -1.6628400022675938 <= q <= 0.6874328810340293)

                     add_NL_constraint(m, :(sigmoid(0.656489652132509 + 0.0160212894179792*sigmoid(-0.2826761707921692 + -0.1840493205106979*$(x[1]) + -0.5336245025582667*$(x[2]) + -0.002104012006349798*$(x[3]) + -0.2812445893570059*$(x[4]) + -0.4922320944855678*$(x[5])) + -0.09426427963418194*sigmoid(0.86904138244507 + 0.8205100119827042*$(x[1]) + -0.9974262299909729*$(x[2]) + -0.6629668417623802*$(x[3]) + 0.6001177079415165*$(x[4]) + -0.3330871079458433*$(x[5]))) + sigmoid(-0.856890340196617 + 0.10531849035106422*sigmoid(-0.2826761707921692 + -0.1840493205106979*$(x[1]) + -0.5336245025582667*$(x[2]) + -0.002104012006349798*$(x[3]) + -0.2812445893570059*$(x[4]) + -0.4922320944855678*$(x[5])) + -0.19686455882338993*sigmoid(0.86904138244507 + 0.8205100119827042*$(x[1]) + -0.9974262299909729*$(x[2]) + -0.6629668417623802*$(x[3]) + 0.6001177079415165*$(x[4]) + -0.3330871079458433*$(x[5]))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    