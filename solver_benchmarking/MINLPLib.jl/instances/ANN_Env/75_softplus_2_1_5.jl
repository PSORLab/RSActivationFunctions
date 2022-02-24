using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -6.7172365869297215 <= q <= 3.8657529446129524)

                     add_NL_constraint(m, :(softplus(0.10971879992509148 + -0.31290714990830937*$(x[1]) + 0.781487131205628*$(x[2])) + softplus(-0.5708088154317625 + 0.5853291463969379*$(x[1]) + -0.9903406373931247*$(x[2])) + softplus(0.5809119298414265 + 0.5024686961828384*$(x[1]) + -0.4096226237713658*$(x[2])) + softplus(-0.6435921083509291 + 0.52943737688555*$(x[1]) + 0.2727617072630939*$(x[2])) + softplus(-0.9019716271422107 + 0.4041768804112231*$(x[1]) + 0.5029634163532655*$(x[2])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    