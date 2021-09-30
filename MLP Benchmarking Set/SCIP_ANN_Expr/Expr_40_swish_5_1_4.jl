using JuMP, EAGO

                     m = Model()

                     register(m, :tswish, 1, tswish, autodiff = true)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -9.337911800929497 <= q <= 10.516730504063752)

                     add_NL_constraint(m, :(tswish(0.2343289771305117 + 0.8628141960120974*$(x[1]) + -0.11862931215058792*$(x[2]) + -0.7515792770646397*$(x[3]) + -0.07887504235853715*$(x[4]) + -0.028101875823106415*$(x[5])) + tswish(-0.795163818920047 + 0.8592483676784317*$(x[1]) + 0.29834699121869246*$(x[2]) + -0.2512699403772065*$(x[3]) + -0.6789499780069534*$(x[4]) + 0.692428517494267*$(x[5])) + tswish(0.4535446921729971 + 0.42858799780567436*$(x[1]) + -0.8440943194398147*$(x[2]) + -0.9767319696599257*$(x[3]) + -0.36754693492627055*$(x[4]) + -0.24489144799480078*$(x[5])) + tswish(0.6966995011836654 + -0.5408975850790565*$(x[1]) + -0.46484786291434776*$(x[2]) + -0.4211464289017286*$(x[3]) + -0.6581786754730552*$(x[4]) + 0.36015443211742815*$(x[5])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    