using JuMP, EAGO

                     m = Model()

                     register(m, :tgelu, 1, tgelu, autodiff = true)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -9.17438870307719 <= q <= 12.307108658713094)

                     add_NL_constraint(m, :(tgelu(0.7344641153473308 + -0.23692570017293724*tgelu(0.34901021581929026 + -0.12939217960265026*tgelu(-0.675958533194557 + -0.5640189184089768*$(x[1]) + -0.559243472495214*$(x[2]) + -0.7027061587782106*$(x[3])) + 0.5852265967345636*tgelu(0.801993103360874 + 0.8590784194050207*$(x[1]) + -0.6627067523268404*$(x[2]) + 0.20053486392302*$(x[3])) + -0.3483623185067346*tgelu(-0.2983672609585555 + 0.9090385846081124*$(x[1]) + 0.13603214513430917*$(x[2]) + -0.5847988467222507*$(x[3]))) + -0.22854131689857704*tgelu(-0.5939599332388568 + 0.09318964735439739*tgelu(-0.675958533194557 + -0.5640189184089768*$(x[1]) + -0.559243472495214*$(x[2]) + -0.7027061587782106*$(x[3])) + 0.5979860361070624*tgelu(0.801993103360874 + 0.8590784194050207*$(x[1]) + -0.6627067523268404*$(x[2]) + 0.20053486392302*$(x[3])) + -0.8670748257994676*tgelu(-0.2983672609585555 + 0.9090385846081124*$(x[1]) + 0.13603214513430917*$(x[2]) + -0.5847988467222507*$(x[3]))) + -0.6959568086086536*tgelu(0.06374644653656736 + 0.45381514084626984*tgelu(-0.675958533194557 + -0.5640189184089768*$(x[1]) + -0.559243472495214*$(x[2]) + -0.7027061587782106*$(x[3])) + 0.9680768262109423*tgelu(0.801993103360874 + 0.8590784194050207*$(x[1]) + -0.6627067523268404*$(x[2]) + 0.20053486392302*$(x[3])) + -0.18015003360300152*tgelu(-0.2983672609585555 + 0.9090385846081124*$(x[1]) + 0.13603214513430917*$(x[2]) + -0.5847988467222507*$(x[3])))) + tgelu(0.6530169062915343 + -0.20184971245217698*tgelu(0.34901021581929026 + -0.12939217960265026*tgelu(-0.675958533194557 + -0.5640189184089768*$(x[1]) + -0.559243472495214*$(x[2]) + -0.7027061587782106*$(x[3])) + 0.5852265967345636*tgelu(0.801993103360874 + 0.8590784194050207*$(x[1]) + -0.6627067523268404*$(x[2]) + 0.20053486392302*$(x[3])) + -0.3483623185067346*tgelu(-0.2983672609585555 + 0.9090385846081124*$(x[1]) + 0.13603214513430917*$(x[2]) + -0.5847988467222507*$(x[3]))) + -0.4970254411199875*tgelu(-0.5939599332388568 + 0.09318964735439739*tgelu(-0.675958533194557 + -0.5640189184089768*$(x[1]) + -0.559243472495214*$(x[2]) + -0.7027061587782106*$(x[3])) + 0.5979860361070624*tgelu(0.801993103360874 + 0.8590784194050207*$(x[1]) + -0.6627067523268404*$(x[2]) + 0.20053486392302*$(x[3])) + -0.8670748257994676*tgelu(-0.2983672609585555 + 0.9090385846081124*$(x[1]) + 0.13603214513430917*$(x[2]) + -0.5847988467222507*$(x[3]))) + -0.751360932433982*tgelu(0.06374644653656736 + 0.45381514084626984*tgelu(-0.675958533194557 + -0.5640189184089768*$(x[1]) + -0.559243472495214*$(x[2]) + -0.7027061587782106*$(x[3])) + 0.9680768262109423*tgelu(0.801993103360874 + 0.8590784194050207*$(x[1]) + -0.6627067523268404*$(x[2]) + 0.20053486392302*$(x[3])) + -0.18015003360300152*tgelu(-0.2983672609585555 + 0.9090385846081124*$(x[1]) + 0.13603214513430917*$(x[2]) + -0.5847988467222507*$(x[3])))) + tgelu(0.4016027128109636 + 0.2900103106142815*tgelu(0.34901021581929026 + -0.12939217960265026*tgelu(-0.675958533194557 + -0.5640189184089768*$(x[1]) + -0.559243472495214*$(x[2]) + -0.7027061587782106*$(x[3])) + 0.5852265967345636*tgelu(0.801993103360874 + 0.8590784194050207*$(x[1]) + -0.6627067523268404*$(x[2]) + 0.20053486392302*$(x[3])) + -0.3483623185067346*tgelu(-0.2983672609585555 + 0.9090385846081124*$(x[1]) + 0.13603214513430917*$(x[2]) + -0.5847988467222507*$(x[3]))) + -0.5082986835414749*tgelu(-0.5939599332388568 + 0.09318964735439739*tgelu(-0.675958533194557 + -0.5640189184089768*$(x[1]) + -0.559243472495214*$(x[2]) + -0.7027061587782106*$(x[3])) + 0.5979860361070624*tgelu(0.801993103360874 + 0.8590784194050207*$(x[1]) + -0.6627067523268404*$(x[2]) + 0.20053486392302*$(x[3])) + -0.8670748257994676*tgelu(-0.2983672609585555 + 0.9090385846081124*$(x[1]) + 0.13603214513430917*$(x[2]) + -0.5847988467222507*$(x[3]))) + -0.17209553226315855*tgelu(0.06374644653656736 + 0.45381514084626984*tgelu(-0.675958533194557 + -0.5640189184089768*$(x[1]) + -0.559243472495214*$(x[2]) + -0.7027061587782106*$(x[3])) + 0.9680768262109423*tgelu(0.801993103360874 + 0.8590784194050207*$(x[1]) + -0.6627067523268404*$(x[2]) + 0.20053486392302*$(x[3])) + -0.18015003360300152*tgelu(-0.2983672609585555 + 0.9090385846081124*$(x[1]) + 0.13603214513430917*$(x[2]) + -0.5847988467222507*$(x[3])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    