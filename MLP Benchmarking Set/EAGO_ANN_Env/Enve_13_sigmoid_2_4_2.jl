using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -0.24111991764255591 <= q <= 1.1768115009705933)

                     add_NL_constraint(m, :(sigmoid(0.1613440592818116 + -0.5603683705176188*sigmoid(-0.5551988621892017 + 0.9062675243946323*sigmoid(0.8854786877492775 + 0.027502661086596714*sigmoid(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5473031829046704*sigmoid(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2]))) + -0.951769937876342*sigmoid(0.7681837460598508 + -0.5022498059206524*sigmoid(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5788103410139427*sigmoid(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2])))) + 0.7198819300515327*sigmoid(-0.5364335362168204 + -0.0070377083880242*sigmoid(0.8854786877492775 + 0.027502661086596714*sigmoid(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5473031829046704*sigmoid(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2]))) + -0.5288062015024475*sigmoid(0.7681837460598508 + -0.5022498059206524*sigmoid(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5788103410139427*sigmoid(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2]))))) + sigmoid(0.9515964690060663 + -0.3348628972770227*sigmoid(-0.5551988621892017 + 0.9062675243946323*sigmoid(0.8854786877492775 + 0.027502661086596714*sigmoid(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5473031829046704*sigmoid(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2]))) + -0.951769937876342*sigmoid(0.7681837460598508 + -0.5022498059206524*sigmoid(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5788103410139427*sigmoid(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2])))) + 0.4586267272558642*sigmoid(-0.5364335362168204 + -0.0070377083880242*sigmoid(0.8854786877492775 + 0.027502661086596714*sigmoid(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5473031829046704*sigmoid(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2]))) + -0.5288062015024475*sigmoid(0.7681837460598508 + -0.5022498059206524*sigmoid(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5788103410139427*sigmoid(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2]))))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    