using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -17.39516631142554 <= q <= 12.050938789121952)

                     add_NL_constraint(m, :(softplus(0.1892165831624366 + -0.6797303891528506*softplus(0.23382152176968107 + 0.970568112692761*$(x[1]) + 0.9147961292868287*$(x[2]) + 0.25099364821370385*$(x[3])) + -0.5563706493225857*softplus(0.1824369939040409 + -0.636682724457545*$(x[1]) + 0.684450108697257*$(x[2]) + -0.3896097552087272*$(x[3])) + -0.5881166064998418*softplus(-0.7017137443246066 + -0.579761469455264*$(x[1]) + -0.38090529589576505*$(x[2]) + -0.30474275998045197*$(x[3])) + -0.5324749908347566*softplus(0.495407339331571 + -0.11045663155169017*$(x[1]) + -0.19556503402268088*$(x[2]) + 0.8751710578699647*$(x[3]))) + softplus(-0.7087227741366906 + -0.3339546766337098*softplus(0.23382152176968107 + 0.970568112692761*$(x[1]) + 0.9147961292868287*$(x[2]) + 0.25099364821370385*$(x[3])) + -0.8987724666577948*softplus(0.1824369939040409 + -0.636682724457545*$(x[1]) + 0.684450108697257*$(x[2]) + -0.3896097552087272*$(x[3])) + 0.3907505378456695*softplus(-0.7017137443246066 + -0.579761469455264*$(x[1]) + -0.38090529589576505*$(x[2]) + -0.30474275998045197*$(x[3])) + -0.7983638213649877*softplus(0.495407339331571 + -0.11045663155169017*$(x[1]) + -0.19556503402268088*$(x[2]) + 0.8751710578699647*$(x[3]))) + softplus(0.16171979073463483 + 0.31171990946083294*softplus(0.23382152176968107 + 0.970568112692761*$(x[1]) + 0.9147961292868287*$(x[2]) + 0.25099364821370385*$(x[3])) + 0.2757921127232579*softplus(0.1824369939040409 + -0.636682724457545*$(x[1]) + 0.684450108697257*$(x[2]) + -0.3896097552087272*$(x[3])) + -0.732639957443523*softplus(-0.7017137443246066 + -0.579761469455264*$(x[1]) + -0.38090529589576505*$(x[2]) + -0.30474275998045197*$(x[3])) + -0.947011017876227*softplus(0.495407339331571 + -0.11045663155169017*$(x[1]) + -0.19556503402268088*$(x[2]) + 0.8751710578699647*$(x[3]))) + softplus(-0.17479526548317725 + -0.7121643152938684*softplus(0.23382152176968107 + 0.970568112692761*$(x[1]) + 0.9147961292868287*$(x[2]) + 0.25099364821370385*$(x[3])) + -0.20573796119512977*softplus(0.1824369939040409 + -0.636682724457545*$(x[1]) + 0.684450108697257*$(x[2]) + -0.3896097552087272*$(x[3])) + 0.920351043144791*softplus(-0.7017137443246066 + -0.579761469455264*$(x[1]) + -0.38090529589576505*$(x[2]) + -0.30474275998045197*$(x[3])) + -0.8770524767395065*softplus(0.495407339331571 + -0.11045663155169017*$(x[1]) + -0.19556503402268088*$(x[2]) + 0.8751710578699647*$(x[3]))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    