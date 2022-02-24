using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -8.232182974914569 <= q <= 6.070970235703473)

                     add_NL_constraint(m, :(1/(1 + exp(-(-0.17705543456567296 + 0.5634748900178774*1/(1 + exp(-(-0.9628319588335761 + 0.5826659544327617*$(x[1]) + -0.6099922924008303*$(x[2]) + 0.9101904633653426*$(x[3])))) + 0.6106029608527352*1/(1 + exp(-(0.06629760484384972 + -0.7340864295918403*$(x[1]) + 0.2309908240081593*$(x[2]) + 0.7867070707759196*$(x[3])))) + 0.5384388086098779*1/(1 + exp(-(-0.07370832879656497 + -0.32073040453400337*$(x[1]) + -0.07649911788086428*$(x[2]) + 0.03499506906851524*$(x[3]))))))) + 1/(1 + exp(-(0.08053767290922664 + 0.911477397877237*1/(1 + exp(-(-0.9628319588335761 + 0.5826659544327617*$(x[1]) + -0.6099922924008303*$(x[2]) + 0.9101904633653426*$(x[3])))) + 0.7134526118957916*1/(1 + exp(-(0.06629760484384972 + -0.7340864295918403*$(x[1]) + 0.2309908240081593*$(x[2]) + 0.7867070707759196*$(x[3])))) + -0.34097520379571256*1/(1 + exp(-(-0.07370832879656497 + -0.32073040453400337*$(x[1]) + -0.07649911788086428*$(x[2]) + 0.03499506906851524*$(x[3]))))))) + 1/(1 + exp(-(0.5992621446321471 + 0.15304126986204158*1/(1 + exp(-(-0.9628319588335761 + 0.5826659544327617*$(x[1]) + -0.6099922924008303*$(x[2]) + 0.9101904633653426*$(x[3])))) + -0.37134199071135*1/(1 + exp(-(0.06629760484384972 + -0.7340864295918403*$(x[1]) + 0.2309908240081593*$(x[2]) + 0.7867070707759196*$(x[3])))) + 0.8747239793986465*1/(1 + exp(-(-0.07370832879656497 + -0.32073040453400337*$(x[1]) + -0.07649911788086428*$(x[2]) + 0.03499506906851524*$(x[3]))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    