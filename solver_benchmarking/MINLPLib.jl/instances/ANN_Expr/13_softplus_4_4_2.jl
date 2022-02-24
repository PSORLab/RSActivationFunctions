using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -2.9922365365158297 <= q <= 1.3959867501971397)

                     add_NL_constraint(m, :(log(1 + exp(-0.7172866311917714 + 0.8530094465195428*log(1 + exp(0.49975691401319544 + 0.817850285997455*log(1 + exp(-0.9044530867886094 + -0.3504940249833268*log(1 + exp(0.9239332442221326 + 0.41157080299867177*$(x[1]) + 0.6856618842893334*$(x[2]) + 0.23250449051531152*$(x[3]) + -0.8171094215290706*$(x[4]))) + -0.925387475816613*log(1 + exp(-0.1441575648301785 + -0.7324773919132506*$(x[1]) + 0.1666248349273194*$(x[2]) + 0.5034013066229974*$(x[3]) + -0.3351866390106042*$(x[4]))))) + -0.15011340339301915*log(1 + exp(-0.49866226232151334 + 0.31623027699891626*log(1 + exp(0.9239332442221326 + 0.41157080299867177*$(x[1]) + 0.6856618842893334*$(x[2]) + 0.23250449051531152*$(x[3]) + -0.8171094215290706*$(x[4]))) + 0.1823838704363938*log(1 + exp(-0.1441575648301785 + -0.7324773919132506*$(x[1]) + 0.1666248349273194*$(x[2]) + 0.5034013066229974*$(x[3]) + -0.3351866390106042*$(x[4]))))))) + -0.2302663292873568*log(1 + exp(0.6056562038214484 + 0.017530899400252764*log(1 + exp(-0.9044530867886094 + -0.3504940249833268*log(1 + exp(0.9239332442221326 + 0.41157080299867177*$(x[1]) + 0.6856618842893334*$(x[2]) + 0.23250449051531152*$(x[3]) + -0.8171094215290706*$(x[4]))) + -0.925387475816613*log(1 + exp(-0.1441575648301785 + -0.7324773919132506*$(x[1]) + 0.1666248349273194*$(x[2]) + 0.5034013066229974*$(x[3]) + -0.3351866390106042*$(x[4]))))) + 0.9141090581590978*log(1 + exp(-0.49866226232151334 + 0.31623027699891626*log(1 + exp(0.9239332442221326 + 0.41157080299867177*$(x[1]) + 0.6856618842893334*$(x[2]) + 0.23250449051531152*$(x[3]) + -0.8171094215290706*$(x[4]))) + 0.1823838704363938*log(1 + exp(-0.1441575648301785 + -0.7324773919132506*$(x[1]) + 0.1666248349273194*$(x[2]) + 0.5034013066229974*$(x[3]) + -0.3351866390106042*$(x[4]))))))))) + log(1 + exp(0.11802092343800519 + -0.23198020992542823*log(1 + exp(0.49975691401319544 + 0.817850285997455*log(1 + exp(-0.9044530867886094 + -0.3504940249833268*log(1 + exp(0.9239332442221326 + 0.41157080299867177*$(x[1]) + 0.6856618842893334*$(x[2]) + 0.23250449051531152*$(x[3]) + -0.8171094215290706*$(x[4]))) + -0.925387475816613*log(1 + exp(-0.1441575648301785 + -0.7324773919132506*$(x[1]) + 0.1666248349273194*$(x[2]) + 0.5034013066229974*$(x[3]) + -0.3351866390106042*$(x[4]))))) + -0.15011340339301915*log(1 + exp(-0.49866226232151334 + 0.31623027699891626*log(1 + exp(0.9239332442221326 + 0.41157080299867177*$(x[1]) + 0.6856618842893334*$(x[2]) + 0.23250449051531152*$(x[3]) + -0.8171094215290706*$(x[4]))) + 0.1823838704363938*log(1 + exp(-0.1441575648301785 + -0.7324773919132506*$(x[1]) + 0.1666248349273194*$(x[2]) + 0.5034013066229974*$(x[3]) + -0.3351866390106042*$(x[4]))))))) + 0.3694431087223533*log(1 + exp(0.6056562038214484 + 0.017530899400252764*log(1 + exp(-0.9044530867886094 + -0.3504940249833268*log(1 + exp(0.9239332442221326 + 0.41157080299867177*$(x[1]) + 0.6856618842893334*$(x[2]) + 0.23250449051531152*$(x[3]) + -0.8171094215290706*$(x[4]))) + -0.925387475816613*log(1 + exp(-0.1441575648301785 + -0.7324773919132506*$(x[1]) + 0.1666248349273194*$(x[2]) + 0.5034013066229974*$(x[3]) + -0.3351866390106042*$(x[4]))))) + 0.9141090581590978*log(1 + exp(-0.49866226232151334 + 0.31623027699891626*log(1 + exp(0.9239332442221326 + 0.41157080299867177*$(x[1]) + 0.6856618842893334*$(x[2]) + 0.23250449051531152*$(x[3]) + -0.8171094215290706*$(x[4]))) + 0.1823838704363938*log(1 + exp(-0.1441575648301785 + -0.7324773919132506*$(x[1]) + 0.1666248349273194*$(x[2]) + 0.5034013066229974*$(x[3]) + -0.3351866390106042*$(x[4]))))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    