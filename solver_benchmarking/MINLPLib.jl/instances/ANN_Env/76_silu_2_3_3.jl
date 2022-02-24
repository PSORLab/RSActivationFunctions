using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -4.847398418942631 <= q <= 3.5461678827819525)

                     add_NL_constraint(m, :(swish(-0.19745953189550702 + -0.698583212677272*swish(-0.2024495957518173 + -0.1501654471281335*swish(-0.06497322351233636 + -0.4876409540129627*$(x[1]) + -0.026006048535915927*$(x[2])) + 0.7965619538518913*swish(-0.4201092923480956 + 0.6085672502473995*$(x[1]) + -0.465188513706837*$(x[2])) + 0.8656395810315112*swish(-0.9600965095983032 + 0.022609326108050443*$(x[1]) + -0.12508836215483532*$(x[2]))) + 0.07925279857906231*swish(-0.4774715210476401 + -0.18966514690787006*swish(-0.06497322351233636 + -0.4876409540129627*$(x[1]) + -0.026006048535915927*$(x[2])) + 0.1687755600264067*swish(-0.4201092923480956 + 0.6085672502473995*$(x[1]) + -0.465188513706837*$(x[2])) + 0.9882336248352352*swish(-0.9600965095983032 + 0.022609326108050443*$(x[1]) + -0.12508836215483532*$(x[2]))) + -0.8658353105986021*swish(-0.5967554189497073 + 0.7632066186026765*swish(-0.06497322351233636 + -0.4876409540129627*$(x[1]) + -0.026006048535915927*$(x[2])) + -0.767873838532454*swish(-0.4201092923480956 + 0.6085672502473995*$(x[1]) + -0.465188513706837*$(x[2])) + 0.02410968604346264*swish(-0.9600965095983032 + 0.022609326108050443*$(x[1]) + -0.12508836215483532*$(x[2])))) + swish(0.9047173820312868 + 0.09375520961098927*swish(-0.2024495957518173 + -0.1501654471281335*swish(-0.06497322351233636 + -0.4876409540129627*$(x[1]) + -0.026006048535915927*$(x[2])) + 0.7965619538518913*swish(-0.4201092923480956 + 0.6085672502473995*$(x[1]) + -0.465188513706837*$(x[2])) + 0.8656395810315112*swish(-0.9600965095983032 + 0.022609326108050443*$(x[1]) + -0.12508836215483532*$(x[2]))) + -0.9282166334997988*swish(-0.4774715210476401 + -0.18966514690787006*swish(-0.06497322351233636 + -0.4876409540129627*$(x[1]) + -0.026006048535915927*$(x[2])) + 0.1687755600264067*swish(-0.4201092923480956 + 0.6085672502473995*$(x[1]) + -0.465188513706837*$(x[2])) + 0.9882336248352352*swish(-0.9600965095983032 + 0.022609326108050443*$(x[1]) + -0.12508836215483532*$(x[2]))) + 0.3928411969784107*swish(-0.5967554189497073 + 0.7632066186026765*swish(-0.06497322351233636 + -0.4876409540129627*$(x[1]) + -0.026006048535915927*$(x[2])) + -0.767873838532454*swish(-0.4201092923480956 + 0.6085672502473995*$(x[1]) + -0.465188513706837*$(x[2])) + 0.02410968604346264*swish(-0.9600965095983032 + 0.022609326108050443*$(x[1]) + -0.12508836215483532*$(x[2])))) + swish(0.16193188153327842 + 0.4964335766578394*swish(-0.2024495957518173 + -0.1501654471281335*swish(-0.06497322351233636 + -0.4876409540129627*$(x[1]) + -0.026006048535915927*$(x[2])) + 0.7965619538518913*swish(-0.4201092923480956 + 0.6085672502473995*$(x[1]) + -0.465188513706837*$(x[2])) + 0.8656395810315112*swish(-0.9600965095983032 + 0.022609326108050443*$(x[1]) + -0.12508836215483532*$(x[2]))) + -0.6165233713312808*swish(-0.4774715210476401 + -0.18966514690787006*swish(-0.06497322351233636 + -0.4876409540129627*$(x[1]) + -0.026006048535915927*$(x[2])) + 0.1687755600264067*swish(-0.4201092923480956 + 0.6085672502473995*$(x[1]) + -0.465188513706837*$(x[2])) + 0.9882336248352352*swish(-0.9600965095983032 + 0.022609326108050443*$(x[1]) + -0.12508836215483532*$(x[2]))) + -0.4911670927879337*swish(-0.5967554189497073 + 0.7632066186026765*swish(-0.06497322351233636 + -0.4876409540129627*$(x[1]) + -0.026006048535915927*$(x[2])) + -0.767873838532454*swish(-0.4201092923480956 + 0.6085672502473995*$(x[1]) + -0.465188513706837*$(x[2])) + 0.02410968604346264*swish(-0.9600965095983032 + 0.022609326108050443*$(x[1]) + -0.12508836215483532*$(x[2])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    