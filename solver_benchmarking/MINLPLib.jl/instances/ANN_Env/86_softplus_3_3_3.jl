using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -16.957145096415093 <= q <= 17.94995800016443)

                     add_NL_constraint(m, :(softplus(0.510608367415756 + 0.13576703913191768*softplus(-0.023366318069798275 + -0.6972645804796413*softplus(-0.6865389686816434 + 0.7515314314281274*$(x[1]) + -0.6334263786214325*$(x[2]) + -0.023277189078302563*$(x[3])) + 0.623708383159082*softplus(0.5286761878087685 + -0.9557352407199864*$(x[1]) + -0.7485274707208927*$(x[2]) + -0.13145677322460392*$(x[3])) + -0.5526763214522168*softplus(0.7131777426657733 + -0.00917591460146161*$(x[1]) + -0.5192676904312425*$(x[2]) + 0.597536428647615*$(x[3]))) + 0.3585964327428499*softplus(-0.17574028037161238 + -0.933468976339102*softplus(-0.6865389686816434 + 0.7515314314281274*$(x[1]) + -0.6334263786214325*$(x[2]) + -0.023277189078302563*$(x[3])) + -0.824606977579692*softplus(0.5286761878087685 + -0.9557352407199864*$(x[1]) + -0.7485274707208927*$(x[2]) + -0.13145677322460392*$(x[3])) + -0.3761458682864913*softplus(0.7131777426657733 + -0.00917591460146161*$(x[1]) + -0.5192676904312425*$(x[2]) + 0.597536428647615*$(x[3]))) + 0.2511872166134532*softplus(0.9643857894710246 + 0.9348568947078797*softplus(-0.6865389686816434 + 0.7515314314281274*$(x[1]) + -0.6334263786214325*$(x[2]) + -0.023277189078302563*$(x[3])) + -0.4027229019076284*softplus(0.5286761878087685 + -0.9557352407199864*$(x[1]) + -0.7485274707208927*$(x[2]) + -0.13145677322460392*$(x[3])) + -0.6455757662896615*softplus(0.7131777426657733 + -0.00917591460146161*$(x[1]) + -0.5192676904312425*$(x[2]) + 0.597536428647615*$(x[3])))) + softplus(-0.7493006135688143 + 0.2483293387413994*softplus(-0.023366318069798275 + -0.6972645804796413*softplus(-0.6865389686816434 + 0.7515314314281274*$(x[1]) + -0.6334263786214325*$(x[2]) + -0.023277189078302563*$(x[3])) + 0.623708383159082*softplus(0.5286761878087685 + -0.9557352407199864*$(x[1]) + -0.7485274707208927*$(x[2]) + -0.13145677322460392*$(x[3])) + -0.5526763214522168*softplus(0.7131777426657733 + -0.00917591460146161*$(x[1]) + -0.5192676904312425*$(x[2]) + 0.597536428647615*$(x[3]))) + -0.7456041150082044*softplus(-0.17574028037161238 + -0.933468976339102*softplus(-0.6865389686816434 + 0.7515314314281274*$(x[1]) + -0.6334263786214325*$(x[2]) + -0.023277189078302563*$(x[3])) + -0.824606977579692*softplus(0.5286761878087685 + -0.9557352407199864*$(x[1]) + -0.7485274707208927*$(x[2]) + -0.13145677322460392*$(x[3])) + -0.3761458682864913*softplus(0.7131777426657733 + -0.00917591460146161*$(x[1]) + -0.5192676904312425*$(x[2]) + 0.597536428647615*$(x[3]))) + 0.49415154189847854*softplus(0.9643857894710246 + 0.9348568947078797*softplus(-0.6865389686816434 + 0.7515314314281274*$(x[1]) + -0.6334263786214325*$(x[2]) + -0.023277189078302563*$(x[3])) + -0.4027229019076284*softplus(0.5286761878087685 + -0.9557352407199864*$(x[1]) + -0.7485274707208927*$(x[2]) + -0.13145677322460392*$(x[3])) + -0.6455757662896615*softplus(0.7131777426657733 + -0.00917591460146161*$(x[1]) + -0.5192676904312425*$(x[2]) + 0.597536428647615*$(x[3])))) + softplus(0.3105901915386422 + 0.5384893864274902*softplus(-0.023366318069798275 + -0.6972645804796413*softplus(-0.6865389686816434 + 0.7515314314281274*$(x[1]) + -0.6334263786214325*$(x[2]) + -0.023277189078302563*$(x[3])) + 0.623708383159082*softplus(0.5286761878087685 + -0.9557352407199864*$(x[1]) + -0.7485274707208927*$(x[2]) + -0.13145677322460392*$(x[3])) + -0.5526763214522168*softplus(0.7131777426657733 + -0.00917591460146161*$(x[1]) + -0.5192676904312425*$(x[2]) + 0.597536428647615*$(x[3]))) + 0.5405413918049242*softplus(-0.17574028037161238 + -0.933468976339102*softplus(-0.6865389686816434 + 0.7515314314281274*$(x[1]) + -0.6334263786214325*$(x[2]) + -0.023277189078302563*$(x[3])) + -0.824606977579692*softplus(0.5286761878087685 + -0.9557352407199864*$(x[1]) + -0.7485274707208927*$(x[2]) + -0.13145677322460392*$(x[3])) + -0.3761458682864913*softplus(0.7131777426657733 + -0.00917591460146161*$(x[1]) + -0.5192676904312425*$(x[2]) + 0.597536428647615*$(x[3]))) + -0.18067211353899282*softplus(0.9643857894710246 + 0.9348568947078797*softplus(-0.6865389686816434 + 0.7515314314281274*$(x[1]) + -0.6334263786214325*$(x[2]) + -0.023277189078302563*$(x[3])) + -0.4027229019076284*softplus(0.5286761878087685 + -0.9557352407199864*$(x[1]) + -0.7485274707208927*$(x[2]) + -0.13145677322460392*$(x[3])) + -0.6455757662896615*softplus(0.7131777426657733 + -0.00917591460146161*$(x[1]) + -0.5192676904312425*$(x[2]) + 0.597536428647615*$(x[3])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    