using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -21.863410525560596 <= q <= 16.759760876289164)

                     add_NL_constraint(m, :(sigmoid(0.9581056824210887 + -0.8972601624172669*sigmoid(0.48456652255491806 + -0.9479094643931347*$(x[1]) + 0.18152729751871233*$(x[2]) + 0.7384779276945834*$(x[3])) + -0.07659521037650441*sigmoid(0.4745256168805634 + -0.6381307282045885*$(x[1]) + 0.08195516705272388*$(x[2]) + -0.876257975110236*$(x[3])) + -0.00831342918317901*sigmoid(-0.5782265433361609 + -0.09311478032414255*$(x[1]) + 0.3276758381100211*$(x[2]) + -0.5571519977300197*$(x[3])) + 0.1691979199232172*sigmoid(0.26331670491475423 + -0.3676524725110859*$(x[1]) + 0.7966590207410333*$(x[2]) + -0.9932629681830947*$(x[3])) + 0.2250938939434315*sigmoid(0.386010874972309 + -0.3966180473443468*$(x[1]) + 0.798506297871798*$(x[2]) + -0.8898772959394607*$(x[3]))) + sigmoid(0.2690390484943683 + 0.08374067856749612*sigmoid(0.48456652255491806 + -0.9479094643931347*$(x[1]) + 0.18152729751871233*$(x[2]) + 0.7384779276945834*$(x[3])) + -0.6533303261729757*sigmoid(0.4745256168805634 + -0.6381307282045885*$(x[1]) + 0.08195516705272388*$(x[2]) + -0.876257975110236*$(x[3])) + 0.840995626029025*sigmoid(-0.5782265433361609 + -0.09311478032414255*$(x[1]) + 0.3276758381100211*$(x[2]) + -0.5571519977300197*$(x[3])) + -0.7082531400429768*sigmoid(0.26331670491475423 + -0.3676524725110859*$(x[1]) + 0.7966590207410333*$(x[2]) + -0.9932629681830947*$(x[3])) + 0.06951275270138302*sigmoid(0.386010874972309 + -0.3966180473443468*$(x[1]) + 0.798506297871798*$(x[2]) + -0.8898772959394607*$(x[3]))) + sigmoid(0.11587500955241081 + 0.3754704068995882*sigmoid(0.48456652255491806 + -0.9479094643931347*$(x[1]) + 0.18152729751871233*$(x[2]) + 0.7384779276945834*$(x[3])) + -0.191407839151704*sigmoid(0.4745256168805634 + -0.6381307282045885*$(x[1]) + 0.08195516705272388*$(x[2]) + -0.876257975110236*$(x[3])) + -0.4785586124200374*sigmoid(-0.5782265433361609 + -0.09311478032414255*$(x[1]) + 0.3276758381100211*$(x[2]) + -0.5571519977300197*$(x[3])) + -0.25460717842408664*sigmoid(0.26331670491475423 + -0.3676524725110859*$(x[1]) + 0.7966590207410333*$(x[2]) + -0.9932629681830947*$(x[3])) + -0.782921189468055*sigmoid(0.386010874972309 + -0.3966180473443468*$(x[1]) + 0.798506297871798*$(x[2]) + -0.8898772959394607*$(x[3]))) + sigmoid(-0.4824105902730653 + -0.3013752487327923*sigmoid(0.48456652255491806 + -0.9479094643931347*$(x[1]) + 0.18152729751871233*$(x[2]) + 0.7384779276945834*$(x[3])) + -0.9808933849306238*sigmoid(0.4745256168805634 + -0.6381307282045885*$(x[1]) + 0.08195516705272388*$(x[2]) + -0.876257975110236*$(x[3])) + 0.29886869236116986*sigmoid(-0.5782265433361609 + -0.09311478032414255*$(x[1]) + 0.3276758381100211*$(x[2]) + -0.5571519977300197*$(x[3])) + -0.6223230896474554*sigmoid(0.26331670491475423 + -0.3676524725110859*$(x[1]) + 0.7966590207410333*$(x[2]) + -0.9932629681830947*$(x[3])) + -0.08733562649470361*sigmoid(0.386010874972309 + -0.3966180473443468*$(x[1]) + 0.798506297871798*$(x[2]) + -0.8898772959394607*$(x[3]))) + sigmoid(-0.5029749667979058 + -0.5112736688682915*sigmoid(0.48456652255491806 + -0.9479094643931347*$(x[1]) + 0.18152729751871233*$(x[2]) + 0.7384779276945834*$(x[3])) + 0.20104536050837662*sigmoid(0.4745256168805634 + -0.6381307282045885*$(x[1]) + 0.08195516705272388*$(x[2]) + -0.876257975110236*$(x[3])) + -0.14474810240563096*sigmoid(-0.5782265433361609 + -0.09311478032414255*$(x[1]) + 0.3276758381100211*$(x[2]) + -0.5571519977300197*$(x[3])) + -0.9195165063894657*sigmoid(0.26331670491475423 + -0.3676524725110859*$(x[1]) + 0.7966590207410333*$(x[2]) + -0.9932629681830947*$(x[3])) + -0.9458134814437154*sigmoid(0.386010874972309 + -0.3966180473443468*$(x[1]) + 0.798506297871798*$(x[2]) + -0.8898772959394607*$(x[3]))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    