using JuMP, EAGO

                     m = Model()

                     register(m, :tsoftplus, 1, tsoftplus, autodiff = true)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -27.478598830779923 <= q <= 33.01357140505985)

                     add_NL_constraint(m, :(tsoftplus(-0.6617868998192131 + 0.9492946915208895*tsoftplus(-0.03820269334683646 + -0.6863611552479676*tsoftplus(0.3361264053192965 + -0.5821215077810429*$(x[1]) + 0.4292828731630256*$(x[2]) + -0.8598290811729683*$(x[3]) + 0.8748227271281661*$(x[4]) + -0.3988413795717971*$(x[5])) + 0.007547446083281173*tsoftplus(-0.10395513200178286 + -0.015850161103474925*$(x[1]) + 0.6361807869411895*$(x[2]) + 0.3939833537669277*$(x[3]) + -0.4542294136097973*$(x[4]) + 0.18712494364193732*$(x[5])) + -0.9918152064719665*tsoftplus(0.7145783933133716 + -0.3039452649712979*$(x[1]) + -0.6482680824148934*$(x[2]) + 0.2675172243573556*$(x[3]) + 0.9038355824308715*$(x[4]) + -0.8495587986892716*$(x[5]))) + 0.5533486487904917*tsoftplus(-0.6941736073421403 + -0.9793252104423442*tsoftplus(0.3361264053192965 + -0.5821215077810429*$(x[1]) + 0.4292828731630256*$(x[2]) + -0.8598290811729683*$(x[3]) + 0.8748227271281661*$(x[4]) + -0.3988413795717971*$(x[5])) + -0.7413330648008167*tsoftplus(-0.10395513200178286 + -0.015850161103474925*$(x[1]) + 0.6361807869411895*$(x[2]) + 0.3939833537669277*$(x[3]) + -0.4542294136097973*$(x[4]) + 0.18712494364193732*$(x[5])) + 0.9204053952440661*tsoftplus(0.7145783933133716 + -0.3039452649712979*$(x[1]) + -0.6482680824148934*$(x[2]) + 0.2675172243573556*$(x[3]) + 0.9038355824308715*$(x[4]) + -0.8495587986892716*$(x[5]))) + 0.6750453910549936*tsoftplus(0.6624124108467315 + -0.8075496605493613*tsoftplus(0.3361264053192965 + -0.5821215077810429*$(x[1]) + 0.4292828731630256*$(x[2]) + -0.8598290811729683*$(x[3]) + 0.8748227271281661*$(x[4]) + -0.3988413795717971*$(x[5])) + -0.2952545653848748*tsoftplus(-0.10395513200178286 + -0.015850161103474925*$(x[1]) + 0.6361807869411895*$(x[2]) + 0.3939833537669277*$(x[3]) + -0.4542294136097973*$(x[4]) + 0.18712494364193732*$(x[5])) + -0.43831909723863216*tsoftplus(0.7145783933133716 + -0.3039452649712979*$(x[1]) + -0.6482680824148934*$(x[2]) + 0.2675172243573556*$(x[3]) + 0.9038355824308715*$(x[4]) + -0.8495587986892716*$(x[5])))) + tsoftplus(0.005291113849781759 + -0.9980293787159806*tsoftplus(-0.03820269334683646 + -0.6863611552479676*tsoftplus(0.3361264053192965 + -0.5821215077810429*$(x[1]) + 0.4292828731630256*$(x[2]) + -0.8598290811729683*$(x[3]) + 0.8748227271281661*$(x[4]) + -0.3988413795717971*$(x[5])) + 0.007547446083281173*tsoftplus(-0.10395513200178286 + -0.015850161103474925*$(x[1]) + 0.6361807869411895*$(x[2]) + 0.3939833537669277*$(x[3]) + -0.4542294136097973*$(x[4]) + 0.18712494364193732*$(x[5])) + -0.9918152064719665*tsoftplus(0.7145783933133716 + -0.3039452649712979*$(x[1]) + -0.6482680824148934*$(x[2]) + 0.2675172243573556*$(x[3]) + 0.9038355824308715*$(x[4]) + -0.8495587986892716*$(x[5]))) + 0.46757760868582476*tsoftplus(-0.6941736073421403 + -0.9793252104423442*tsoftplus(0.3361264053192965 + -0.5821215077810429*$(x[1]) + 0.4292828731630256*$(x[2]) + -0.8598290811729683*$(x[3]) + 0.8748227271281661*$(x[4]) + -0.3988413795717971*$(x[5])) + -0.7413330648008167*tsoftplus(-0.10395513200178286 + -0.015850161103474925*$(x[1]) + 0.6361807869411895*$(x[2]) + 0.3939833537669277*$(x[3]) + -0.4542294136097973*$(x[4]) + 0.18712494364193732*$(x[5])) + 0.9204053952440661*tsoftplus(0.7145783933133716 + -0.3039452649712979*$(x[1]) + -0.6482680824148934*$(x[2]) + 0.2675172243573556*$(x[3]) + 0.9038355824308715*$(x[4]) + -0.8495587986892716*$(x[5]))) + -0.5871650576572001*tsoftplus(0.6624124108467315 + -0.8075496605493613*tsoftplus(0.3361264053192965 + -0.5821215077810429*$(x[1]) + 0.4292828731630256*$(x[2]) + -0.8598290811729683*$(x[3]) + 0.8748227271281661*$(x[4]) + -0.3988413795717971*$(x[5])) + -0.2952545653848748*tsoftplus(-0.10395513200178286 + -0.015850161103474925*$(x[1]) + 0.6361807869411895*$(x[2]) + 0.3939833537669277*$(x[3]) + -0.4542294136097973*$(x[4]) + 0.18712494364193732*$(x[5])) + -0.43831909723863216*tsoftplus(0.7145783933133716 + -0.3039452649712979*$(x[1]) + -0.6482680824148934*$(x[2]) + 0.2675172243573556*$(x[3]) + 0.9038355824308715*$(x[4]) + -0.8495587986892716*$(x[5])))) + tsoftplus(0.7624336166284693 + 0.09632042161741294*tsoftplus(-0.03820269334683646 + -0.6863611552479676*tsoftplus(0.3361264053192965 + -0.5821215077810429*$(x[1]) + 0.4292828731630256*$(x[2]) + -0.8598290811729683*$(x[3]) + 0.8748227271281661*$(x[4]) + -0.3988413795717971*$(x[5])) + 0.007547446083281173*tsoftplus(-0.10395513200178286 + -0.015850161103474925*$(x[1]) + 0.6361807869411895*$(x[2]) + 0.3939833537669277*$(x[3]) + -0.4542294136097973*$(x[4]) + 0.18712494364193732*$(x[5])) + -0.9918152064719665*tsoftplus(0.7145783933133716 + -0.3039452649712979*$(x[1]) + -0.6482680824148934*$(x[2]) + 0.2675172243573556*$(x[3]) + 0.9038355824308715*$(x[4]) + -0.8495587986892716*$(x[5]))) + 0.4884862603615341*tsoftplus(-0.6941736073421403 + -0.9793252104423442*tsoftplus(0.3361264053192965 + -0.5821215077810429*$(x[1]) + 0.4292828731630256*$(x[2]) + -0.8598290811729683*$(x[3]) + 0.8748227271281661*$(x[4]) + -0.3988413795717971*$(x[5])) + -0.7413330648008167*tsoftplus(-0.10395513200178286 + -0.015850161103474925*$(x[1]) + 0.6361807869411895*$(x[2]) + 0.3939833537669277*$(x[3]) + -0.4542294136097973*$(x[4]) + 0.18712494364193732*$(x[5])) + 0.9204053952440661*tsoftplus(0.7145783933133716 + -0.3039452649712979*$(x[1]) + -0.6482680824148934*$(x[2]) + 0.2675172243573556*$(x[3]) + 0.9038355824308715*$(x[4]) + -0.8495587986892716*$(x[5]))) + -0.4778431596724957*tsoftplus(0.6624124108467315 + -0.8075496605493613*tsoftplus(0.3361264053192965 + -0.5821215077810429*$(x[1]) + 0.4292828731630256*$(x[2]) + -0.8598290811729683*$(x[3]) + 0.8748227271281661*$(x[4]) + -0.3988413795717971*$(x[5])) + -0.2952545653848748*tsoftplus(-0.10395513200178286 + -0.015850161103474925*$(x[1]) + 0.6361807869411895*$(x[2]) + 0.3939833537669277*$(x[3]) + -0.4542294136097973*$(x[4]) + 0.18712494364193732*$(x[5])) + -0.43831909723863216*tsoftplus(0.7145783933133716 + -0.3039452649712979*$(x[1]) + -0.6482680824148934*$(x[2]) + 0.2675172243573556*$(x[3]) + 0.9038355824308715*$(x[4]) + -0.8495587986892716*$(x[5])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    