using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -2.5130750055538225 <= q <= 0.5461809853160992)

                     add_NL_constraint(m, :(sigmoid(0.2725479121754346 + -0.8300423816078824*sigmoid(0.4883764676450131 + -0.839508705198635*sigmoid(-0.21991358432134245 + 0.30382799975157004*$(x[1]) + -0.38537831260498834*$(x[2]) + -0.1290291387191438*$(x[3]) + 0.10002178041754783*$(x[4])) + -0.2973781388226713*sigmoid(0.2827046831263944 + 0.5752082862528085*$(x[1]) + -0.012653071875058153*$(x[2]) + 0.6838325139346542*$(x[3]) + -0.2881145069121702*$(x[4]))) + -0.12971248788862733*sigmoid(0.6069520335591818 + -0.2920681483245766*sigmoid(-0.21991358432134245 + 0.30382799975157004*$(x[1]) + -0.38537831260498834*$(x[2]) + -0.1290291387191438*$(x[3]) + 0.10002178041754783*$(x[4])) + -0.06339008805133695*sigmoid(0.2827046831263944 + 0.5752082862528085*$(x[1]) + -0.012653071875058153*$(x[2]) + 0.6838325139346542*$(x[3]) + -0.2881145069121702*$(x[4])))) + sigmoid(-0.3539044424622513 + -0.8729033532715484*sigmoid(0.4883764676450131 + -0.839508705198635*sigmoid(-0.21991358432134245 + 0.30382799975157004*$(x[1]) + -0.38537831260498834*$(x[2]) + -0.1290291387191438*$(x[3]) + 0.10002178041754783*$(x[4])) + -0.2973781388226713*sigmoid(0.2827046831263944 + 0.5752082862528085*$(x[1]) + -0.012653071875058153*$(x[2]) + 0.6838325139346542*$(x[3]) + -0.2881145069121702*$(x[4]))) + -0.8220211155852892*sigmoid(0.6069520335591818 + -0.2920681483245766*sigmoid(-0.21991358432134245 + 0.30382799975157004*$(x[1]) + -0.38537831260498834*$(x[2]) + -0.1290291387191438*$(x[3]) + 0.10002178041754783*$(x[4])) + -0.06339008805133695*sigmoid(0.2827046831263944 + 0.5752082862528085*$(x[1]) + -0.012653071875058153*$(x[2]) + 0.6838325139346542*$(x[3]) + -0.2881145069121702*$(x[4])))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    