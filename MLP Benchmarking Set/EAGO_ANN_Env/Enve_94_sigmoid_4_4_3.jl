using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -25.333057560042874 <= q <= 25.572942595086705)

                     add_NL_constraint(m, :(sigmoid(0.5431424517802732 + 0.3215464175845826*sigmoid(-0.992329408744888 + 0.035315499984154286*sigmoid(-0.39773055792249634 + -0.2688382045333171*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.14794021485851117*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + -0.16789058859301598*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + 0.7354012698161392*sigmoid(-0.9405430101509649 + 0.8514527517929098*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.4556174656696874*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.7882988135289852*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + -0.07451801785017942*sigmoid(-0.7654670188752783 + 0.9592935842609438*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + -0.9322865866827668*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.36003604402525236*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4])))) + 0.48752332481162597*sigmoid(0.013513144221706241 + 0.8731099271834082*sigmoid(-0.39773055792249634 + -0.2688382045333171*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.14794021485851117*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + -0.16789058859301598*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + 0.10929449794292001*sigmoid(-0.9405430101509649 + 0.8514527517929098*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.4556174656696874*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.7882988135289852*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + -0.5376764022807836*sigmoid(-0.7654670188752783 + 0.9592935842609438*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + -0.9322865866827668*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.36003604402525236*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4])))) + -0.882307385651925*sigmoid(0.47205877925539896 + -0.7275153977061217*sigmoid(-0.39773055792249634 + -0.2688382045333171*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.14794021485851117*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + -0.16789058859301598*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + -0.06942737222989992*sigmoid(-0.9405430101509649 + 0.8514527517929098*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.4556174656696874*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.7882988135289852*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + 0.7700347182674769*sigmoid(-0.7654670188752783 + 0.9592935842609438*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + -0.9322865866827668*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.36003604402525236*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))))) + sigmoid(-0.41301601406626487 + -0.6420491546454796*sigmoid(-0.992329408744888 + 0.035315499984154286*sigmoid(-0.39773055792249634 + -0.2688382045333171*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.14794021485851117*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + -0.16789058859301598*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + 0.7354012698161392*sigmoid(-0.9405430101509649 + 0.8514527517929098*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.4556174656696874*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.7882988135289852*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + -0.07451801785017942*sigmoid(-0.7654670188752783 + 0.9592935842609438*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + -0.9322865866827668*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.36003604402525236*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4])))) + -0.6144852208996268*sigmoid(0.013513144221706241 + 0.8731099271834082*sigmoid(-0.39773055792249634 + -0.2688382045333171*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.14794021485851117*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + -0.16789058859301598*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + 0.10929449794292001*sigmoid(-0.9405430101509649 + 0.8514527517929098*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.4556174656696874*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.7882988135289852*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + -0.5376764022807836*sigmoid(-0.7654670188752783 + 0.9592935842609438*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + -0.9322865866827668*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.36003604402525236*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4])))) + 0.2444814377311353*sigmoid(0.47205877925539896 + -0.7275153977061217*sigmoid(-0.39773055792249634 + -0.2688382045333171*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.14794021485851117*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + -0.16789058859301598*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + -0.06942737222989992*sigmoid(-0.9405430101509649 + 0.8514527517929098*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.4556174656696874*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.7882988135289852*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + 0.7700347182674769*sigmoid(-0.7654670188752783 + 0.9592935842609438*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + -0.9322865866827668*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.36003604402525236*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))))) + sigmoid(0.24170146483025556 + 0.6355468155095405*sigmoid(-0.992329408744888 + 0.035315499984154286*sigmoid(-0.39773055792249634 + -0.2688382045333171*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.14794021485851117*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + -0.16789058859301598*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + 0.7354012698161392*sigmoid(-0.9405430101509649 + 0.8514527517929098*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.4556174656696874*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.7882988135289852*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + -0.07451801785017942*sigmoid(-0.7654670188752783 + 0.9592935842609438*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + -0.9322865866827668*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.36003604402525236*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4])))) + -0.4679259068500996*sigmoid(0.013513144221706241 + 0.8731099271834082*sigmoid(-0.39773055792249634 + -0.2688382045333171*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.14794021485851117*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + -0.16789058859301598*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + 0.10929449794292001*sigmoid(-0.9405430101509649 + 0.8514527517929098*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.4556174656696874*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.7882988135289852*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + -0.5376764022807836*sigmoid(-0.7654670188752783 + 0.9592935842609438*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + -0.9322865866827668*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.36003604402525236*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4])))) + -0.4322274059671325*sigmoid(0.47205877925539896 + -0.7275153977061217*sigmoid(-0.39773055792249634 + -0.2688382045333171*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.14794021485851117*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + -0.16789058859301598*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + -0.06942737222989992*sigmoid(-0.9405430101509649 + 0.8514527517929098*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + 0.4556174656696874*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.7882988135289852*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))) + 0.7700347182674769*sigmoid(-0.7654670188752783 + 0.9592935842609438*sigmoid(0.26665022299511376 + 0.01156522854721187*$(x[1]) + -0.6964846692837381*$(x[2]) + -0.7191400279556142*$(x[3]) + 0.43955560132837324*$(x[4])) + -0.9322865866827668*sigmoid(0.486199564399207 + 0.68175366633267*$(x[1]) + 0.9228207924687095*$(x[2]) + 0.9906941832773559*$(x[3]) + -0.15966562493733782*$(x[4])) + 0.36003604402525236*sigmoid(-0.4576254501765571 + -0.19080935940515786*$(x[1]) + 0.6266529061684065*$(x[2]) + 0.821004395317809*$(x[3]) + -0.8077485321867957*$(x[4]))))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    