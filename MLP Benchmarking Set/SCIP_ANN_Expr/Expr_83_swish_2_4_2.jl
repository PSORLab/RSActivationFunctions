using JuMP, EAGO

                     m = Model()

                     register(m, :tswish, 1, tswish, autodiff = true)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -1.204321297033384 <= q <= 0.8243408974838059)

                     add_NL_constraint(m, :(tswish(-0.3091096341671835 + -0.8618941840632734*tswish(-0.6209180702342523 + -0.0973590097759951*tswish(-0.04047838100610157 + 0.4621156425151769*tswish(0.11503449382870645 + -0.7621431199433788*$(x[1]) + -0.4850634019611211*$(x[2])) + -0.8841495159309423*tswish(0.570414859211203 + 0.6458312973604934*$(x[1]) + -0.9655941779327*$(x[2]))) + 0.603009880096415*tswish(-0.453281655466772 + -0.11223110736185182*tswish(0.11503449382870645 + -0.7621431199433788*$(x[1]) + -0.4850634019611211*$(x[2])) + 0.14880789591097976*tswish(0.570414859211203 + 0.6458312973604934*$(x[1]) + -0.9655941779327*$(x[2])))) + 0.5188745940457764*tswish(-0.20069289498327114 + 0.7913830070586334*tswish(-0.04047838100610157 + 0.4621156425151769*tswish(0.11503449382870645 + -0.7621431199433788*$(x[1]) + -0.4850634019611211*$(x[2])) + -0.8841495159309423*tswish(0.570414859211203 + 0.6458312973604934*$(x[1]) + -0.9655941779327*$(x[2]))) + 0.580856834117863*tswish(-0.453281655466772 + -0.11223110736185182*tswish(0.11503449382870645 + -0.7621431199433788*$(x[1]) + -0.4850634019611211*$(x[2])) + 0.14880789591097976*tswish(0.570414859211203 + 0.6458312973604934*$(x[1]) + -0.9655941779327*$(x[2]))))) + tswish(0.1491769777235712 + 0.3688804868411881*tswish(-0.6209180702342523 + -0.0973590097759951*tswish(-0.04047838100610157 + 0.4621156425151769*tswish(0.11503449382870645 + -0.7621431199433788*$(x[1]) + -0.4850634019611211*$(x[2])) + -0.8841495159309423*tswish(0.570414859211203 + 0.6458312973604934*$(x[1]) + -0.9655941779327*$(x[2]))) + 0.603009880096415*tswish(-0.453281655466772 + -0.11223110736185182*tswish(0.11503449382870645 + -0.7621431199433788*$(x[1]) + -0.4850634019611211*$(x[2])) + 0.14880789591097976*tswish(0.570414859211203 + 0.6458312973604934*$(x[1]) + -0.9655941779327*$(x[2])))) + -0.8963811448749599*tswish(-0.20069289498327114 + 0.7913830070586334*tswish(-0.04047838100610157 + 0.4621156425151769*tswish(0.11503449382870645 + -0.7621431199433788*$(x[1]) + -0.4850634019611211*$(x[2])) + -0.8841495159309423*tswish(0.570414859211203 + 0.6458312973604934*$(x[1]) + -0.9655941779327*$(x[2]))) + 0.580856834117863*tswish(-0.453281655466772 + -0.11223110736185182*tswish(0.11503449382870645 + -0.7621431199433788*$(x[1]) + -0.4850634019611211*$(x[2])) + 0.14880789591097976*tswish(0.570414859211203 + 0.6458312973604934*$(x[1]) + -0.9655941779327*$(x[2]))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    