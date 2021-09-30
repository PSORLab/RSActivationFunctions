using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -13.394230108685145 <= q <= 10.288345149293914)

                     add_NL_constraint(m, :(swish(-0.6965252935719177 + -0.4925100713909978*swish(-0.2012255509021914 + -0.32018890002217804*$(x[1]) + 0.030272995813719117*$(x[2])) + 0.44086046165311554*swish(0.7112215171266847 + -0.7885998193642538*$(x[1]) + 0.167087022795124*$(x[2])) + 0.7650963190972195*swish(-0.9601918651039938 + 0.9704084740657772*$(x[1]) + -0.6984284303471284*$(x[2])) + 0.8289547516454809*swish(0.5529850935936662 + 0.3560396632363272*$(x[1]) + 0.29783445549836474*$(x[2])) + -0.042858330032117475*swish(0.37368777277589205 + 0.6249595351606145*$(x[1]) + 0.7326998121945953*$(x[2]))) + swish(-0.6024724484259911 + 0.2426183187517128*swish(-0.2012255509021914 + -0.32018890002217804*$(x[1]) + 0.030272995813719117*$(x[2])) + 0.01950505383380774*swish(0.7112215171266847 + -0.7885998193642538*$(x[1]) + 0.167087022795124*$(x[2])) + 0.3213888306389019*swish(-0.9601918651039938 + 0.9704084740657772*$(x[1]) + -0.6984284303471284*$(x[2])) + -0.9164758290037409*swish(0.5529850935936662 + 0.3560396632363272*$(x[1]) + 0.29783445549836474*$(x[2])) + -0.6587049020883917*swish(0.37368777277589205 + 0.6249595351606145*$(x[1]) + 0.7326998121945953*$(x[2]))) + swish(-0.4975741047281068 + -0.5627723609443134*swish(-0.2012255509021914 + -0.32018890002217804*$(x[1]) + 0.030272995813719117*$(x[2])) + 0.27098872130046736*swish(0.7112215171266847 + -0.7885998193642538*$(x[1]) + 0.167087022795124*$(x[2])) + -0.8275860880512909*swish(-0.9601918651039938 + 0.9704084740657772*$(x[1]) + -0.6984284303471284*$(x[2])) + 0.6550769042138529*swish(0.5529850935936662 + 0.3560396632363272*$(x[1]) + 0.29783445549836474*$(x[2])) + -0.04870507794637735*swish(0.37368777277589205 + 0.6249595351606145*$(x[1]) + 0.7326998121945953*$(x[2]))) + swish(-0.7992727846265533 + -0.02419952186865082*swish(-0.2012255509021914 + -0.32018890002217804*$(x[1]) + 0.030272995813719117*$(x[2])) + 0.36808466108005344*swish(0.7112215171266847 + -0.7885998193642538*$(x[1]) + 0.167087022795124*$(x[2])) + 0.282207884797264*swish(-0.9601918651039938 + 0.9704084740657772*$(x[1]) + -0.6984284303471284*$(x[2])) + 0.34549380281477404*swish(0.5529850935936662 + 0.3560396632363272*$(x[1]) + 0.29783445549836474*$(x[2])) + 0.5845630457596838*swish(0.37368777277589205 + 0.6249595351606145*$(x[1]) + 0.7326998121945953*$(x[2]))) + swish(0.5865219690095511 + 0.4619489075483818*swish(-0.2012255509021914 + -0.32018890002217804*$(x[1]) + 0.030272995813719117*$(x[2])) + 0.6118927168725068*swish(0.7112215171266847 + -0.7885998193642538*$(x[1]) + 0.167087022795124*$(x[2])) + 0.9935053538715097*swish(-0.9601918651039938 + 0.9704084740657772*$(x[1]) + -0.6984284303471284*$(x[2])) + 0.041722664030451995*swish(0.5529850935936662 + 0.3560396632363272*$(x[1]) + 0.29783445549836474*$(x[2])) + 0.45832734429761723*swish(0.37368777277589205 + 0.6249595351606145*$(x[1]) + 0.7326998121945953*$(x[2]))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    