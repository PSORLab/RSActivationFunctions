using JuMP, EAGO

                     m = Model()

                     register(m, :tgelu, 1, tgelu, autodiff = true)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -0.24111991764255591 <= q <= 1.1768115009705933)

                     add_NL_constraint(m, :(tgelu(0.1613440592818116 + -0.5603683705176188*tgelu(-0.5551988621892017 + 0.9062675243946323*tgelu(0.8854786877492775 + 0.027502661086596714*tgelu(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5473031829046704*tgelu(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2]))) + -0.951769937876342*tgelu(0.7681837460598508 + -0.5022498059206524*tgelu(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5788103410139427*tgelu(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2])))) + 0.7198819300515327*tgelu(-0.5364335362168204 + -0.0070377083880242*tgelu(0.8854786877492775 + 0.027502661086596714*tgelu(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5473031829046704*tgelu(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2]))) + -0.5288062015024475*tgelu(0.7681837460598508 + -0.5022498059206524*tgelu(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5788103410139427*tgelu(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2]))))) + tgelu(0.9515964690060663 + -0.3348628972770227*tgelu(-0.5551988621892017 + 0.9062675243946323*tgelu(0.8854786877492775 + 0.027502661086596714*tgelu(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5473031829046704*tgelu(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2]))) + -0.951769937876342*tgelu(0.7681837460598508 + -0.5022498059206524*tgelu(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5788103410139427*tgelu(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2])))) + 0.4586267272558642*tgelu(-0.5364335362168204 + -0.0070377083880242*tgelu(0.8854786877492775 + 0.027502661086596714*tgelu(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5473031829046704*tgelu(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2]))) + -0.5288062015024475*tgelu(0.7681837460598508 + -0.5022498059206524*tgelu(0.4360807668660964 + -0.37320228113341125*$(x[1]) + -0.09462609625480844*$(x[2])) + 0.5788103410139427*tgelu(0.005870932675498874 + 0.23033076905194427*$(x[1]) + 0.3263768103406788*$(x[2]))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    