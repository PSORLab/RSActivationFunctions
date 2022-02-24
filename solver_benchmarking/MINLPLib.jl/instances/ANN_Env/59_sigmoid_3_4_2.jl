using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -4.116064765345723 <= q <= -0.12537054293327043)

                     add_NL_constraint(m, :(sigmoid(-0.6620590692147021 + -0.9580027962881332*sigmoid(-0.38090481240177176 + -0.02939531016329644*sigmoid(0.2067810723024852 + 0.7469716803367321*sigmoid(-0.2319263638689142 + 0.46965024077813977*$(x[1]) + 0.028636456496167195*$(x[2]) + 0.15542595955334626*$(x[3])) + -0.3209180228281623*sigmoid(-0.2352871682017641 + -0.6225824541901428*$(x[1]) + -0.2700435280904796*$(x[2]) + -0.7791025613764826*$(x[3]))) + -0.9724733578803808*sigmoid(-0.4274287128450034 + -0.22403315015562253*sigmoid(-0.2319263638689142 + 0.46965024077813977*$(x[1]) + 0.028636456496167195*$(x[2]) + 0.15542595955334626*$(x[3])) + -0.6136110647248842*sigmoid(-0.2352871682017641 + -0.6225824541901428*$(x[1]) + -0.2700435280904796*$(x[2]) + -0.7791025613764826*$(x[3])))) + -0.699289524220764*sigmoid(0.4498619401955626 + 0.5463276179102481*sigmoid(0.2067810723024852 + 0.7469716803367321*sigmoid(-0.2319263638689142 + 0.46965024077813977*$(x[1]) + 0.028636456496167195*$(x[2]) + 0.15542595955334626*$(x[3])) + -0.3209180228281623*sigmoid(-0.2352871682017641 + -0.6225824541901428*$(x[1]) + -0.2700435280904796*$(x[2]) + -0.7791025613764826*$(x[3]))) + -0.02415027804893688*sigmoid(-0.4274287128450034 + -0.22403315015562253*sigmoid(-0.2319263638689142 + 0.46965024077813977*$(x[1]) + 0.028636456496167195*$(x[2]) + 0.15542595955334626*$(x[3])) + -0.6136110647248842*sigmoid(-0.2352871682017641 + -0.6225824541901428*$(x[1]) + -0.2700435280904796*$(x[2]) + -0.7791025613764826*$(x[3]))))) + sigmoid(-0.8107501452143855 + 0.008470632510253484*sigmoid(-0.38090481240177176 + -0.02939531016329644*sigmoid(0.2067810723024852 + 0.7469716803367321*sigmoid(-0.2319263638689142 + 0.46965024077813977*$(x[1]) + 0.028636456496167195*$(x[2]) + 0.15542595955334626*$(x[3])) + -0.3209180228281623*sigmoid(-0.2352871682017641 + -0.6225824541901428*$(x[1]) + -0.2700435280904796*$(x[2]) + -0.7791025613764826*$(x[3]))) + -0.9724733578803808*sigmoid(-0.4274287128450034 + -0.22403315015562253*sigmoid(-0.2319263638689142 + 0.46965024077813977*$(x[1]) + 0.028636456496167195*$(x[2]) + 0.15542595955334626*$(x[3])) + -0.6136110647248842*sigmoid(-0.2352871682017641 + -0.6225824541901428*$(x[1]) + -0.2700435280904796*$(x[2]) + -0.7791025613764826*$(x[3])))) + 0.39739497887500974*sigmoid(0.4498619401955626 + 0.5463276179102481*sigmoid(0.2067810723024852 + 0.7469716803367321*sigmoid(-0.2319263638689142 + 0.46965024077813977*$(x[1]) + 0.028636456496167195*$(x[2]) + 0.15542595955334626*$(x[3])) + -0.3209180228281623*sigmoid(-0.2352871682017641 + -0.6225824541901428*$(x[1]) + -0.2700435280904796*$(x[2]) + -0.7791025613764826*$(x[3]))) + -0.02415027804893688*sigmoid(-0.4274287128450034 + -0.22403315015562253*sigmoid(-0.2319263638689142 + 0.46965024077813977*$(x[1]) + 0.028636456496167195*$(x[2]) + 0.15542595955334626*$(x[3])) + -0.6136110647248842*sigmoid(-0.2352871682017641 + -0.6225824541901428*$(x[1]) + -0.2700435280904796*$(x[2]) + -0.7791025613764826*$(x[3]))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    