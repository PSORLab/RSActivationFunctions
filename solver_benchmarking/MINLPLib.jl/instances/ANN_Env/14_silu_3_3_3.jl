using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -13.110901270484293 <= q <= 11.380819522292354)

                     add_NL_constraint(m, :(swish(-0.7773677229963925 + -0.31823111196481735*swish(0.13898320039278467 + -0.4573031975765818*swish(-0.1658975405753922 + 0.8948668185465118*$(x[1]) + 0.5677868700589963*$(x[2]) + -0.004745367697401193*$(x[3])) + -0.027363793963717686*swish(0.8084147176169236 + 0.6887886991760568*$(x[1]) + 0.9976411528938542*$(x[2]) + 0.9554800615635051*$(x[3])) + -0.7118578003946046*swish(-0.23665048660081967 + 0.22941639672835912*$(x[1]) + 0.6694807543091978*$(x[2]) + 0.08612932953468189*$(x[3]))) + -0.05869662098445039*swish(0.7177153834386609 + 0.046197114005355644*swish(-0.1658975405753922 + 0.8948668185465118*$(x[1]) + 0.5677868700589963*$(x[2]) + -0.004745367697401193*$(x[3])) + 0.8341427099559482*swish(0.8084147176169236 + 0.6887886991760568*$(x[1]) + 0.9976411528938542*$(x[2]) + 0.9554800615635051*$(x[3])) + -0.5961814570241306*swish(-0.23665048660081967 + 0.22941639672835912*$(x[1]) + 0.6694807543091978*$(x[2]) + 0.08612932953468189*$(x[3]))) + -0.31920840200501477*swish(-0.05621054225465549 + -0.4685096496776473*swish(-0.1658975405753922 + 0.8948668185465118*$(x[1]) + 0.5677868700589963*$(x[2]) + -0.004745367697401193*$(x[3])) + -0.723697578017108*swish(0.8084147176169236 + 0.6887886991760568*$(x[1]) + 0.9976411528938542*$(x[2]) + 0.9554800615635051*$(x[3])) + 0.643520408356284*swish(-0.23665048660081967 + 0.22941639672835912*$(x[1]) + 0.6694807543091978*$(x[2]) + 0.08612932953468189*$(x[3])))) + swish(-0.2800668423712529 + 0.5241807567635033*swish(0.13898320039278467 + -0.4573031975765818*swish(-0.1658975405753922 + 0.8948668185465118*$(x[1]) + 0.5677868700589963*$(x[2]) + -0.004745367697401193*$(x[3])) + -0.027363793963717686*swish(0.8084147176169236 + 0.6887886991760568*$(x[1]) + 0.9976411528938542*$(x[2]) + 0.9554800615635051*$(x[3])) + -0.7118578003946046*swish(-0.23665048660081967 + 0.22941639672835912*$(x[1]) + 0.6694807543091978*$(x[2]) + 0.08612932953468189*$(x[3]))) + 0.9218858277065758*swish(0.7177153834386609 + 0.046197114005355644*swish(-0.1658975405753922 + 0.8948668185465118*$(x[1]) + 0.5677868700589963*$(x[2]) + -0.004745367697401193*$(x[3])) + 0.8341427099559482*swish(0.8084147176169236 + 0.6887886991760568*$(x[1]) + 0.9976411528938542*$(x[2]) + 0.9554800615635051*$(x[3])) + -0.5961814570241306*swish(-0.23665048660081967 + 0.22941639672835912*$(x[1]) + 0.6694807543091978*$(x[2]) + 0.08612932953468189*$(x[3]))) + 0.0068491484591501894*swish(-0.05621054225465549 + -0.4685096496776473*swish(-0.1658975405753922 + 0.8948668185465118*$(x[1]) + 0.5677868700589963*$(x[2]) + -0.004745367697401193*$(x[3])) + -0.723697578017108*swish(0.8084147176169236 + 0.6887886991760568*$(x[1]) + 0.9976411528938542*$(x[2]) + 0.9554800615635051*$(x[3])) + 0.643520408356284*swish(-0.23665048660081967 + 0.22941639672835912*$(x[1]) + 0.6694807543091978*$(x[2]) + 0.08612932953468189*$(x[3])))) + swish(-0.0923150493931284 + -0.18688964829603716*swish(0.13898320039278467 + -0.4573031975765818*swish(-0.1658975405753922 + 0.8948668185465118*$(x[1]) + 0.5677868700589963*$(x[2]) + -0.004745367697401193*$(x[3])) + -0.027363793963717686*swish(0.8084147176169236 + 0.6887886991760568*$(x[1]) + 0.9976411528938542*$(x[2]) + 0.9554800615635051*$(x[3])) + -0.7118578003946046*swish(-0.23665048660081967 + 0.22941639672835912*$(x[1]) + 0.6694807543091978*$(x[2]) + 0.08612932953468189*$(x[3]))) + 0.4699561510424055*swish(0.7177153834386609 + 0.046197114005355644*swish(-0.1658975405753922 + 0.8948668185465118*$(x[1]) + 0.5677868700589963*$(x[2]) + -0.004745367697401193*$(x[3])) + 0.8341427099559482*swish(0.8084147176169236 + 0.6887886991760568*$(x[1]) + 0.9976411528938542*$(x[2]) + 0.9554800615635051*$(x[3])) + -0.5961814570241306*swish(-0.23665048660081967 + 0.22941639672835912*$(x[1]) + 0.6694807543091978*$(x[2]) + 0.08612932953468189*$(x[3]))) + -0.26284635568255643*swish(-0.05621054225465549 + -0.4685096496776473*swish(-0.1658975405753922 + 0.8948668185465118*$(x[1]) + 0.5677868700589963*$(x[2]) + -0.004745367697401193*$(x[3])) + -0.723697578017108*swish(0.8084147176169236 + 0.6887886991760568*$(x[1]) + 0.9976411528938542*$(x[2]) + 0.9554800615635051*$(x[3])) + 0.643520408356284*swish(-0.23665048660081967 + 0.22941639672835912*$(x[1]) + 0.6694807543091978*$(x[2]) + 0.08612932953468189*$(x[3])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    