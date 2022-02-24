using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -8.186501529651974 <= q <= 9.622483160840265)

                     add_NL_constraint(m, :(log(1 + exp(-0.9342689023247241 + -0.06643065035242302*log(1 + exp(-0.6967397920955185 + 0.4612305646899797*log(1 + exp(-0.7651411836742423 + 0.736339264438953*$(x[1]) + 0.06323249947984699*$(x[2]) + 0.6326865446578034*$(x[3]) + 0.8935006364653741*$(x[4]))) + 0.5917923558177942*log(1 + exp(0.8959549878207418 + 0.6412415470262633*$(x[1]) + 0.6383904188579423*$(x[2]) + 0.9706157764666923*$(x[3]) + 0.8108132764672078*$(x[4]))))) + 0.13022424784864484*log(1 + exp(-0.8092681495291991 + -0.6577545463201151*log(1 + exp(-0.7651411836742423 + 0.736339264438953*$(x[1]) + 0.06323249947984699*$(x[2]) + 0.6326865446578034*$(x[3]) + 0.8935006364653741*$(x[4]))) + 0.816846389391447*log(1 + exp(0.8959549878207418 + 0.6412415470262633*$(x[1]) + 0.6383904188579423*$(x[2]) + 0.9706157764666923*$(x[3]) + 0.8108132764672078*$(x[4]))))))) + log(1 + exp(0.9502905692637253 + 0.2731932353707758*log(1 + exp(-0.6967397920955185 + 0.4612305646899797*log(1 + exp(-0.7651411836742423 + 0.736339264438953*$(x[1]) + 0.06323249947984699*$(x[2]) + 0.6326865446578034*$(x[3]) + 0.8935006364653741*$(x[4]))) + 0.5917923558177942*log(1 + exp(0.8959549878207418 + 0.6412415470262633*$(x[1]) + 0.6383904188579423*$(x[2]) + 0.9706157764666923*$(x[3]) + 0.8108132764672078*$(x[4]))))) + 0.7525923835832526*log(1 + exp(-0.8092681495291991 + -0.6577545463201151*log(1 + exp(-0.7651411836742423 + 0.736339264438953*$(x[1]) + 0.06323249947984699*$(x[2]) + 0.6326865446578034*$(x[3]) + 0.8935006364653741*$(x[4]))) + 0.816846389391447*log(1 + exp(0.8959549878207418 + 0.6412415470262633*$(x[1]) + 0.6383904188579423*$(x[2]) + 0.9706157764666923*$(x[3]) + 0.8108132764672078*$(x[4]))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    