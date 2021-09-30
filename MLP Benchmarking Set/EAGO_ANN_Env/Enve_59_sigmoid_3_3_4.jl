using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -25.319850882330265 <= q <= 20.104159599708435)

                     add_NL_constraint(m, :(sigmoid(-0.6888487813469442 + -0.47590614810296294*sigmoid(-0.8830373008649941 + -0.9282613619030653*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + 0.1154590761910046*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + -0.23214870341084914*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + -0.012869674385302954*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3]))) + 0.323742005561948*sigmoid(-0.80732034160591 + 0.4236868055029781*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + -0.8325587735439486*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + -0.45887946418982617*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + -0.9691616168671184*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3]))) + 0.5386501435834856*sigmoid(0.49063371526409405 + 0.15325490505714612*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + -0.28159775042327384*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + 0.0767716680698447*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + -0.372285564791929*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3]))) + -0.6136885274178971*sigmoid(-0.12791111384674858 + 0.8147687041713252*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + -0.8719287991954934*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + -0.4499290661451636*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + 0.6512976998601072*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3])))) + sigmoid(0.22803244920395116 + -0.26482386624083176*sigmoid(-0.8830373008649941 + -0.9282613619030653*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + 0.1154590761910046*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + -0.23214870341084914*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + -0.012869674385302954*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3]))) + 0.3182420852264034*sigmoid(-0.80732034160591 + 0.4236868055029781*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + -0.8325587735439486*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + -0.45887946418982617*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + -0.9691616168671184*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3]))) + -0.6292430172313166*sigmoid(0.49063371526409405 + 0.15325490505714612*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + -0.28159775042327384*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + 0.0767716680698447*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + -0.372285564791929*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3]))) + -0.9510161367687839*sigmoid(-0.12791111384674858 + 0.8147687041713252*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + -0.8719287991954934*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + -0.4499290661451636*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + 0.6512976998601072*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3])))) + sigmoid(-0.36489690117022056 + -0.9285462753849676*sigmoid(-0.8830373008649941 + -0.9282613619030653*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + 0.1154590761910046*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + -0.23214870341084914*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + -0.012869674385302954*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3]))) + 0.7320424772682457*sigmoid(-0.80732034160591 + 0.4236868055029781*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + -0.8325587735439486*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + -0.45887946418982617*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + -0.9691616168671184*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3]))) + 0.15880082897914605*sigmoid(0.49063371526409405 + 0.15325490505714612*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + -0.28159775042327384*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + 0.0767716680698447*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + -0.372285564791929*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3]))) + 0.40146170851310004*sigmoid(-0.12791111384674858 + 0.8147687041713252*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + -0.8719287991954934*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + -0.4499290661451636*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + 0.6512976998601072*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3])))) + sigmoid(-0.2633256361752778 + -0.6466769960843042*sigmoid(-0.8830373008649941 + -0.9282613619030653*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + 0.1154590761910046*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + -0.23214870341084914*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + -0.012869674385302954*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3]))) + 0.8849228202017758*sigmoid(-0.80732034160591 + 0.4236868055029781*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + -0.8325587735439486*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + -0.45887946418982617*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + -0.9691616168671184*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3]))) + -0.29271699202083834*sigmoid(0.49063371526409405 + 0.15325490505714612*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + -0.28159775042327384*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + 0.0767716680698447*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + -0.372285564791929*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3]))) + -0.9939555428124409*sigmoid(-0.12791111384674858 + 0.8147687041713252*sigmoid(0.34957392240454244 + -0.509116247775061*$(x[1]) + 0.18801594250887899*$(x[2]) + 0.9919857061442992*$(x[3])) + -0.8719287991954934*sigmoid(-0.7819402774452504 + 0.9010553090363671*$(x[1]) + 0.824627396192223*$(x[2]) + 0.286712814997784*$(x[3])) + -0.4499290661451636*sigmoid(-0.8232881036646678 + -0.772122469205669*$(x[1]) + -0.13597379372522278*$(x[2]) + 0.6326899168992295*$(x[3])) + 0.6512976998601072*sigmoid(0.7343779954199108 + -0.24028094783101306*$(x[1]) + 0.18963160530909562*$(x[2]) + 0.19095944600140546*$(x[3])))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    