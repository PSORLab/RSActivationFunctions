using JuMP, EAGO

                     m = Model()

                     register(m, :tsigmoid, 1, tsigmoid, autodiff = true)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -7.3812119119199435 <= q <= 10.207462917079203)

                     add_NL_constraint(m, :(tsigmoid(0.5408841267104756 + -0.07837411492616209*$(x[1]) + -0.029680542068924787*$(x[2]) + -0.10666120030580517*$(x[3]) + -0.006765436212655018*$(x[4])) + tsigmoid(0.9338094226762705 + -0.6063994743295322*$(x[1]) + -0.4653767333083927*$(x[2]) + -0.47612332174083116*$(x[3]) + -0.7153413732916332*$(x[4])) + tsigmoid(0.15018119594654378 + 0.37907191806594875*$(x[1]) + 0.37210848547797815*$(x[2]) + 0.4663757045337298*$(x[3]) + 0.736372872506295*$(x[4])) + tsigmoid(0.06578111278585252 + 0.545417540848983*$(x[1]) + -0.8991788016602755*$(x[2]) + -0.7032436157729784*$(x[3]) + 0.7571366614439348*$(x[4])) + tsigmoid(-0.277530355539513 + -0.13574511788370858*$(x[1]) + -0.36417777625697223*$(x[2]) + 0.6018848686740252*$(x[3]) + 0.34890185519080674*$(x[4])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    