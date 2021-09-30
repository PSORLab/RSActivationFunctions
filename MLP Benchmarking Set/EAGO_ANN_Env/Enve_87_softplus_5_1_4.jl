using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -10.191898840964447 <= q <= 13.057268478811217)

                     add_NL_constraint(m, :(softplus(0.3931268107032255 + -0.14417118054881772*$(x[1]) + -0.8983868582487244*$(x[2]) + -0.40555291197016885*$(x[3]) + 0.19017152748205213*$(x[4]) + 0.650793637826117*$(x[5])) + softplus(0.2475513230218267 + 0.46709802598812056*$(x[1]) + -0.542317009512276*$(x[2]) + 0.5875548213914157*$(x[3]) + 0.5059173912567938*$(x[4]) + 0.887951762439974*$(x[5])) + softplus(0.4882595399807288 + -0.42320703512275415*$(x[1]) + 0.515837789259082*$(x[2]) + 0.47053314783722167*$(x[3]) + 0.4712553108093829*$(x[4]) + 0.7191934644286402*$(x[5])) + softplus(0.30374714521760326 + -0.2744691852140897*$(x[1]) + -0.9673459534917677*$(x[2]) + 0.9467371522625538*$(x[3]) + 0.7182830948851877*$(x[4]) + 0.8378063999126906*$(x[5])) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    