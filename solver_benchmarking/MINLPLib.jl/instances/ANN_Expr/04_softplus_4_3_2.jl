using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -0.7481803354345802 <= q <= 3.1247048530266452)

                     add_NL_constraint(m, :(log(1 + exp(0.30283339323429637 + -0.6237495659853991*log(1 + exp(-0.21938578694289834 + -0.903262746252536*log(1 + exp(0.9051056598158431 + -0.13053580855908242*$(x[1]) + -0.5492925144525009*$(x[2]) + 0.019096353213378325*$(x[3]) + 0.3378807953855012*$(x[4]))) + 0.16468819845671145*log(1 + exp(0.06066012774712304 + 0.010865522181132903*$(x[1]) + -0.27940603676198705*$(x[2]) + -0.359831046364532*$(x[3]) + -0.7556261576696839*$(x[4]))))) + 0.9377177734653888*log(1 + exp(0.5819650565710264 + -0.47161154460717736*log(1 + exp(0.9051056598158431 + -0.13053580855908242*$(x[1]) + -0.5492925144525009*$(x[2]) + 0.019096353213378325*$(x[3]) + 0.3378807953855012*$(x[4]))) + -0.23723822344605994*log(1 + exp(0.06066012774712304 + 0.010865522181132903*$(x[1]) + -0.27940603676198705*$(x[2]) + -0.359831046364532*$(x[3]) + -0.7556261576696839*$(x[4]))))))) + log(1 + exp(-0.5162811028857681 + 0.4410139812710443*log(1 + exp(-0.21938578694289834 + -0.903262746252536*log(1 + exp(0.9051056598158431 + -0.13053580855908242*$(x[1]) + -0.5492925144525009*$(x[2]) + 0.019096353213378325*$(x[3]) + 0.3378807953855012*$(x[4]))) + 0.16468819845671145*log(1 + exp(0.06066012774712304 + 0.010865522181132903*$(x[1]) + -0.27940603676198705*$(x[2]) + -0.359831046364532*$(x[3]) + -0.7556261576696839*$(x[4]))))) + 0.2178047619382908*log(1 + exp(0.5819650565710264 + -0.47161154460717736*log(1 + exp(0.9051056598158431 + -0.13053580855908242*$(x[1]) + -0.5492925144525009*$(x[2]) + 0.019096353213378325*$(x[3]) + 0.3378807953855012*$(x[4]))) + -0.23723822344605994*log(1 + exp(0.06066012774712304 + 0.010865522181132903*$(x[1]) + -0.27940603676198705*$(x[2]) + -0.359831046364532*$(x[3]) + -0.7556261576696839*$(x[4]))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    