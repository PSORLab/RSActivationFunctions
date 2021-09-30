using JuMP, EAGO

                     m = Model()

                     register(m, :tgelu, 1, tgelu, autodiff = true)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -7.402963198241181 <= q <= 10.69425184672363)

                     add_NL_constraint(m, :(tgelu(0.8371807044311796 + 0.8704171794028026*tgelu(0.49712926227789733 + -0.8864835702729246*tgelu(0.7622038205205994 + 0.15215956372976613*$(x[1]) + 0.34890864249300835*$(x[2]) + -0.04443105351417209*$(x[3])) + 0.4769399630251083*tgelu(0.9704516310031317 + -0.7273596305474435*$(x[1]) + -0.4620624250873875*$(x[2]) + 0.9788446568288451*$(x[3])) + 0.1497323285201806*tgelu(0.49193125140811267 + -0.23440841092786924*$(x[1]) + -0.7799516854335602*$(x[2]) + 0.27479216706725484*$(x[3]))) + -0.6985780654299014*tgelu(-0.24401262821193903 + 0.47834138771251666*tgelu(0.7622038205205994 + 0.15215956372976613*$(x[1]) + 0.34890864249300835*$(x[2]) + -0.04443105351417209*$(x[3])) + 0.7676518018942731*tgelu(0.9704516310031317 + -0.7273596305474435*$(x[1]) + -0.4620624250873875*$(x[2]) + 0.9788446568288451*$(x[3])) + 0.40016945769328816*tgelu(0.49193125140811267 + -0.23440841092786924*$(x[1]) + -0.7799516854335602*$(x[2]) + 0.27479216706725484*$(x[3]))) + -0.4526412632683501*tgelu(-0.07174032700938637 + -0.8757610419073876*tgelu(0.7622038205205994 + 0.15215956372976613*$(x[1]) + 0.34890864249300835*$(x[2]) + -0.04443105351417209*$(x[3])) + 0.3736787700282149*tgelu(0.9704516310031317 + -0.7273596305474435*$(x[1]) + -0.4620624250873875*$(x[2]) + 0.9788446568288451*$(x[3])) + -0.24119713203949988*tgelu(0.49193125140811267 + -0.23440841092786924*$(x[1]) + -0.7799516854335602*$(x[2]) + 0.27479216706725484*$(x[3])))) + tgelu(0.31617890867709564 + -0.4512906518484101*tgelu(0.49712926227789733 + -0.8864835702729246*tgelu(0.7622038205205994 + 0.15215956372976613*$(x[1]) + 0.34890864249300835*$(x[2]) + -0.04443105351417209*$(x[3])) + 0.4769399630251083*tgelu(0.9704516310031317 + -0.7273596305474435*$(x[1]) + -0.4620624250873875*$(x[2]) + 0.9788446568288451*$(x[3])) + 0.1497323285201806*tgelu(0.49193125140811267 + -0.23440841092786924*$(x[1]) + -0.7799516854335602*$(x[2]) + 0.27479216706725484*$(x[3]))) + -0.525271471569992*tgelu(-0.24401262821193903 + 0.47834138771251666*tgelu(0.7622038205205994 + 0.15215956372976613*$(x[1]) + 0.34890864249300835*$(x[2]) + -0.04443105351417209*$(x[3])) + 0.7676518018942731*tgelu(0.9704516310031317 + -0.7273596305474435*$(x[1]) + -0.4620624250873875*$(x[2]) + 0.9788446568288451*$(x[3])) + 0.40016945769328816*tgelu(0.49193125140811267 + -0.23440841092786924*$(x[1]) + -0.7799516854335602*$(x[2]) + 0.27479216706725484*$(x[3]))) + 0.9634146543115425*tgelu(-0.07174032700938637 + -0.8757610419073876*tgelu(0.7622038205205994 + 0.15215956372976613*$(x[1]) + 0.34890864249300835*$(x[2]) + -0.04443105351417209*$(x[3])) + 0.3736787700282149*tgelu(0.9704516310031317 + -0.7273596305474435*$(x[1]) + -0.4620624250873875*$(x[2]) + 0.9788446568288451*$(x[3])) + -0.24119713203949988*tgelu(0.49193125140811267 + -0.23440841092786924*$(x[1]) + -0.7799516854335602*$(x[2]) + 0.27479216706725484*$(x[3])))) + tgelu(-0.6145076500646258 + -0.4829697882728361*tgelu(0.49712926227789733 + -0.8864835702729246*tgelu(0.7622038205205994 + 0.15215956372976613*$(x[1]) + 0.34890864249300835*$(x[2]) + -0.04443105351417209*$(x[3])) + 0.4769399630251083*tgelu(0.9704516310031317 + -0.7273596305474435*$(x[1]) + -0.4620624250873875*$(x[2]) + 0.9788446568288451*$(x[3])) + 0.1497323285201806*tgelu(0.49193125140811267 + -0.23440841092786924*$(x[1]) + -0.7799516854335602*$(x[2]) + 0.27479216706725484*$(x[3]))) + -0.1668945685815335*tgelu(-0.24401262821193903 + 0.47834138771251666*tgelu(0.7622038205205994 + 0.15215956372976613*$(x[1]) + 0.34890864249300835*$(x[2]) + -0.04443105351417209*$(x[3])) + 0.7676518018942731*tgelu(0.9704516310031317 + -0.7273596305474435*$(x[1]) + -0.4620624250873875*$(x[2]) + 0.9788446568288451*$(x[3])) + 0.40016945769328816*tgelu(0.49193125140811267 + -0.23440841092786924*$(x[1]) + -0.7799516854335602*$(x[2]) + 0.27479216706725484*$(x[3]))) + -0.2147115104392343*tgelu(-0.07174032700938637 + -0.8757610419073876*tgelu(0.7622038205205994 + 0.15215956372976613*$(x[1]) + 0.34890864249300835*$(x[2]) + -0.04443105351417209*$(x[3])) + 0.3736787700282149*tgelu(0.9704516310031317 + -0.7273596305474435*$(x[1]) + -0.4620624250873875*$(x[2]) + 0.9788446568288451*$(x[3])) + -0.24119713203949988*tgelu(0.49193125140811267 + -0.23440841092786924*$(x[1]) + -0.7799516854335602*$(x[2]) + 0.27479216706725484*$(x[3])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    