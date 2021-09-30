using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -28.165921338153836 <= q <= 22.451668718102514)

                     add_NL_constraint(m, :(gelu(0.10101463123951016 + -0.49979102118264107*gelu(0.15804663699401678 + -0.008377069661400505*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.4869894033180575*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.7210855162253278*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.14667345177752633*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.5149443366931137*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + 0.3824835531843225*gelu(-0.7577367905591053 + -0.04166478073890412*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.6938626224785609*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.7782694625261426*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + 0.7778668695583479*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + 0.3821038187715722*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + 0.21411006121231813*gelu(0.3736348207311089 + -0.39208736614049*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + -0.042977531100322874*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.16431789249344497*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + 0.432722266573379*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.4413156803571807*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + -0.12104491193188638*gelu(0.05787464304261469 + 0.5305184613060394*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + -0.6158488051504558*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + -0.23878489761098676*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.9858364192569993*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + 0.3414026426680894*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + -0.9492953256451271*gelu(0.5322965473920065 + -0.7978183700033283*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.9934239882841558*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + -0.11964856103995913*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.2113566972030343*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.43099752350390075*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2])))) + gelu(-0.7610953327324173 + -0.9572436704796683*gelu(0.15804663699401678 + -0.008377069661400505*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.4869894033180575*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.7210855162253278*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.14667345177752633*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.5149443366931137*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + 0.06225604600556389*gelu(-0.7577367905591053 + -0.04166478073890412*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.6938626224785609*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.7782694625261426*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + 0.7778668695583479*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + 0.3821038187715722*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + -0.7449032070249051*gelu(0.3736348207311089 + -0.39208736614049*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + -0.042977531100322874*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.16431789249344497*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + 0.432722266573379*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.4413156803571807*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + -0.5488904885677792*gelu(0.05787464304261469 + 0.5305184613060394*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + -0.6158488051504558*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + -0.23878489761098676*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.9858364192569993*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + 0.3414026426680894*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + -0.04565993863969098*gelu(0.5322965473920065 + -0.7978183700033283*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.9934239882841558*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + -0.11964856103995913*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.2113566972030343*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.43099752350390075*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2])))) + gelu(0.12427055448957525 + -0.027238674917238637*gelu(0.15804663699401678 + -0.008377069661400505*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.4869894033180575*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.7210855162253278*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.14667345177752633*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.5149443366931137*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + 0.308445839080584*gelu(-0.7577367905591053 + -0.04166478073890412*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.6938626224785609*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.7782694625261426*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + 0.7778668695583479*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + 0.3821038187715722*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + -0.8247168793820858*gelu(0.3736348207311089 + -0.39208736614049*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + -0.042977531100322874*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.16431789249344497*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + 0.432722266573379*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.4413156803571807*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + -0.22961988630019992*gelu(0.05787464304261469 + 0.5305184613060394*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + -0.6158488051504558*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + -0.23878489761098676*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.9858364192569993*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + 0.3414026426680894*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + 0.14055972374928638*gelu(0.5322965473920065 + -0.7978183700033283*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.9934239882841558*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + -0.11964856103995913*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.2113566972030343*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.43099752350390075*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2])))) + gelu(0.5310301575214345 + -0.9094852104994371*gelu(0.15804663699401678 + -0.008377069661400505*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.4869894033180575*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.7210855162253278*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.14667345177752633*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.5149443366931137*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + -0.4529087203497979*gelu(-0.7577367905591053 + -0.04166478073890412*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.6938626224785609*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.7782694625261426*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + 0.7778668695583479*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + 0.3821038187715722*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + -0.5349657675722823*gelu(0.3736348207311089 + -0.39208736614049*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + -0.042977531100322874*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.16431789249344497*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + 0.432722266573379*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.4413156803571807*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + 0.6477930619688768*gelu(0.05787464304261469 + 0.5305184613060394*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + -0.6158488051504558*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + -0.23878489761098676*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.9858364192569993*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + 0.3414026426680894*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + 0.45258272606675387*gelu(0.5322965473920065 + -0.7978183700033283*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.9934239882841558*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + -0.11964856103995913*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.2113566972030343*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.43099752350390075*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2])))) + gelu(-0.5790944392440358 + -0.7750781669379663*gelu(0.15804663699401678 + -0.008377069661400505*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.4869894033180575*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.7210855162253278*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.14667345177752633*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.5149443366931137*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + 0.746235174549482*gelu(-0.7577367905591053 + -0.04166478073890412*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.6938626224785609*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.7782694625261426*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + 0.7778668695583479*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + 0.3821038187715722*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + 0.7558395680446419*gelu(0.3736348207311089 + -0.39208736614049*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + -0.042977531100322874*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + 0.16431789249344497*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + 0.432722266573379*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.4413156803571807*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + 0.6886712636449279*gelu(0.05787464304261469 + 0.5305184613060394*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + -0.6158488051504558*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + -0.23878489761098676*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.9858364192569993*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + 0.3414026426680894*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2]))) + 0.6150375155447452*gelu(0.5322965473920065 + -0.7978183700033283*gelu(-0.49167370859453374 + -0.5366642618380912*$(x[1]) + -0.6576968878852809*$(x[2])) + 0.9934239882841558*gelu(-0.7601162686340901 + 0.5747472876670519*$(x[1]) + -0.0063940772459396555*$(x[2])) + -0.11964856103995913*gelu(-0.07774298150552417 + 0.043690035947654415*$(x[1]) + 0.2946388192070528*$(x[2])) + -0.2113566972030343*gelu(-0.6457026825104784 + 0.4644690261065305*$(x[1]) + -0.49854796636509846*$(x[2])) + -0.43099752350390075*gelu(0.5046680112836821 + -0.9495162094032881*$(x[1]) + 0.9985769434282226*$(x[2])))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    