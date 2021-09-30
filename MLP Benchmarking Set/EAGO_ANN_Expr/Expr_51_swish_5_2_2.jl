using JuMP, EAGO

                     m = Model()

                     register(m, :tswish, 1, tswish, autodiff = true)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -1.6628400022675938 <= q <= 0.6874328810340293)

                     add_NL_constraint(m, :(tswish(0.656489652132509 + 0.0160212894179792*tswish(-0.2826761707921692 + -0.1840493205106979*$(x[1]) + -0.5336245025582667*$(x[2]) + -0.002104012006349798*$(x[3]) + -0.2812445893570059*$(x[4]) + -0.4922320944855678*$(x[5])) + -0.09426427963418194*tswish(0.86904138244507 + 0.8205100119827042*$(x[1]) + -0.9974262299909729*$(x[2]) + -0.6629668417623802*$(x[3]) + 0.6001177079415165*$(x[4]) + -0.3330871079458433*$(x[5]))) + tswish(-0.856890340196617 + 0.10531849035106422*tswish(-0.2826761707921692 + -0.1840493205106979*$(x[1]) + -0.5336245025582667*$(x[2]) + -0.002104012006349798*$(x[3]) + -0.2812445893570059*$(x[4]) + -0.4922320944855678*$(x[5])) + -0.19686455882338993*tswish(0.86904138244507 + 0.8205100119827042*$(x[1]) + -0.9974262299909729*$(x[2]) + -0.6629668417623802*$(x[3]) + 0.6001177079415165*$(x[4]) + -0.3330871079458433*$(x[5]))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    