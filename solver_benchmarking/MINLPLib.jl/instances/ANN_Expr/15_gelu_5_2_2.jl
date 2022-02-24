using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -3.517322036224115 <= q <= 6.144134546328734)

                     add_NL_constraint(m, :((0.9006950404367178 + -0.8785024386485549*(0.872987896263127 + 0.13249617278479375*$(x[1]) + 0.7878182386475117*$(x[2]) + -0.4203533662190604*$(x[3]) + -0.7649366056085984*$(x[4]) + -0.57871509222098*$(x[5]))*(1 + erf((0.872987896263127 + 0.13249617278479375*$(x[1]) + 0.7878182386475117*$(x[2]) + -0.4203533662190604*$(x[3]) + -0.7649366056085984*$(x[4]) + -0.57871509222098*$(x[5]))/sqrt(2)))/2 + 0.18543992065441683*(0.8256824299447256 + -0.6033988061517293*$(x[1]) + -0.15324994390362567*$(x[2]) + 0.6933635130851963*$(x[3]) + 0.1238690975426957*$(x[4]) + 0.2325901400223671*$(x[5]))*(1 + erf((0.8256824299447256 + -0.6033988061517293*$(x[1]) + -0.15324994390362567*$(x[2]) + 0.6933635130851963*$(x[3]) + 0.1238690975426957*$(x[4]) + 0.2325901400223671*$(x[5]))/sqrt(2)))/2)*(1 + erf((0.9006950404367178 + -0.8785024386485549*(0.872987896263127 + 0.13249617278479375*$(x[1]) + 0.7878182386475117*$(x[2]) + -0.4203533662190604*$(x[3]) + -0.7649366056085984*$(x[4]) + -0.57871509222098*$(x[5]))*(1 + erf((0.872987896263127 + 0.13249617278479375*$(x[1]) + 0.7878182386475117*$(x[2]) + -0.4203533662190604*$(x[3]) + -0.7649366056085984*$(x[4]) + -0.57871509222098*$(x[5]))/sqrt(2)))/2 + 0.18543992065441683*(0.8256824299447256 + -0.6033988061517293*$(x[1]) + -0.15324994390362567*$(x[2]) + 0.6933635130851963*$(x[3]) + 0.1238690975426957*$(x[4]) + 0.2325901400223671*$(x[5]))*(1 + erf((0.8256824299447256 + -0.6033988061517293*$(x[1]) + -0.15324994390362567*$(x[2]) + 0.6933635130851963*$(x[3]) + 0.1238690975426957*$(x[4]) + 0.2325901400223671*$(x[5]))/sqrt(2)))/2)/sqrt(2)))/2 + (0.10821024944347046 + 0.16585705930469619*(0.872987896263127 + 0.13249617278479375*$(x[1]) + 0.7878182386475117*$(x[2]) + -0.4203533662190604*$(x[3]) + -0.7649366056085984*$(x[4]) + -0.57871509222098*$(x[5]))*(1 + erf((0.872987896263127 + 0.13249617278479375*$(x[1]) + 0.7878182386475117*$(x[2]) + -0.4203533662190604*$(x[3]) + -0.7649366056085984*$(x[4]) + -0.57871509222098*$(x[5]))/sqrt(2)))/2 + 0.9368217650268278*(0.8256824299447256 + -0.6033988061517293*$(x[1]) + -0.15324994390362567*$(x[2]) + 0.6933635130851963*$(x[3]) + 0.1238690975426957*$(x[4]) + 0.2325901400223671*$(x[5]))*(1 + erf((0.8256824299447256 + -0.6033988061517293*$(x[1]) + -0.15324994390362567*$(x[2]) + 0.6933635130851963*$(x[3]) + 0.1238690975426957*$(x[4]) + 0.2325901400223671*$(x[5]))/sqrt(2)))/2)*(1 + erf((0.10821024944347046 + 0.16585705930469619*(0.872987896263127 + 0.13249617278479375*$(x[1]) + 0.7878182386475117*$(x[2]) + -0.4203533662190604*$(x[3]) + -0.7649366056085984*$(x[4]) + -0.57871509222098*$(x[5]))*(1 + erf((0.872987896263127 + 0.13249617278479375*$(x[1]) + 0.7878182386475117*$(x[2]) + -0.4203533662190604*$(x[3]) + -0.7649366056085984*$(x[4]) + -0.57871509222098*$(x[5]))/sqrt(2)))/2 + 0.9368217650268278*(0.8256824299447256 + -0.6033988061517293*$(x[1]) + -0.15324994390362567*$(x[2]) + 0.6933635130851963*$(x[3]) + 0.1238690975426957*$(x[4]) + 0.2325901400223671*$(x[5]))*(1 + erf((0.8256824299447256 + -0.6033988061517293*$(x[1]) + -0.15324994390362567*$(x[2]) + 0.6933635130851963*$(x[3]) + 0.1238690975426957*$(x[4]) + 0.2325901400223671*$(x[5]))/sqrt(2)))/2)/sqrt(2)))/2 - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    