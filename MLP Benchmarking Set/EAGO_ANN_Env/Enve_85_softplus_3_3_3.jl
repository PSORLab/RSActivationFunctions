using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -10.798184106412839 <= q <= 7.8181952856621715)

                     add_NL_constraint(m, :(softplus(-0.3638622169231569 + -0.7894263215435262*softplus(-0.4631148608127349 + -0.7428552262850863*softplus(-0.5341243579019301 + 0.5611324607258696*$(x[1]) + -0.593816232159198*$(x[2]) + 0.34837805869263727*$(x[3])) + -0.05821446851395917*softplus(0.17156635132643272 + 0.07330337401840348*$(x[1]) + -0.17735974291534173*$(x[2]) + -0.9111929848564051*$(x[3])) + -0.035737768406546966*softplus(0.016605037927807764 + 0.8199189360400561*$(x[1]) + 0.588507463268042*$(x[2]) + 0.6043543184008882*$(x[3]))) + -0.3487965070997654*softplus(0.5466854688811069 + 0.5410217402196289*softplus(-0.5341243579019301 + 0.5611324607258696*$(x[1]) + -0.593816232159198*$(x[2]) + 0.34837805869263727*$(x[3])) + -0.43803755847655035*softplus(0.17156635132643272 + 0.07330337401840348*$(x[1]) + -0.17735974291534173*$(x[2]) + -0.9111929848564051*$(x[3])) + 0.6527089787461371*softplus(0.016605037927807764 + 0.8199189360400561*$(x[1]) + 0.588507463268042*$(x[2]) + 0.6043543184008882*$(x[3]))) + -0.07408446700298876*softplus(0.6283785987613308 + 0.5367607999947079*softplus(-0.5341243579019301 + 0.5611324607258696*$(x[1]) + -0.593816232159198*$(x[2]) + 0.34837805869263727*$(x[3])) + -0.7242129606035013*softplus(0.17156635132643272 + 0.07330337401840348*$(x[1]) + -0.17735974291534173*$(x[2]) + -0.9111929848564051*$(x[3])) + 0.6225416211010737*softplus(0.016605037927807764 + 0.8199189360400561*$(x[1]) + 0.588507463268042*$(x[2]) + 0.6043543184008882*$(x[3])))) + softplus(-0.297272833064965 + -0.2682995254754337*softplus(-0.4631148608127349 + -0.7428552262850863*softplus(-0.5341243579019301 + 0.5611324607258696*$(x[1]) + -0.593816232159198*$(x[2]) + 0.34837805869263727*$(x[3])) + -0.05821446851395917*softplus(0.17156635132643272 + 0.07330337401840348*$(x[1]) + -0.17735974291534173*$(x[2]) + -0.9111929848564051*$(x[3])) + -0.035737768406546966*softplus(0.016605037927807764 + 0.8199189360400561*$(x[1]) + 0.588507463268042*$(x[2]) + 0.6043543184008882*$(x[3]))) + 0.14215379587121024*softplus(0.5466854688811069 + 0.5410217402196289*softplus(-0.5341243579019301 + 0.5611324607258696*$(x[1]) + -0.593816232159198*$(x[2]) + 0.34837805869263727*$(x[3])) + -0.43803755847655035*softplus(0.17156635132643272 + 0.07330337401840348*$(x[1]) + -0.17735974291534173*$(x[2]) + -0.9111929848564051*$(x[3])) + 0.6527089787461371*softplus(0.016605037927807764 + 0.8199189360400561*$(x[1]) + 0.588507463268042*$(x[2]) + 0.6043543184008882*$(x[3]))) + 0.987652138244774*softplus(0.6283785987613308 + 0.5367607999947079*softplus(-0.5341243579019301 + 0.5611324607258696*$(x[1]) + -0.593816232159198*$(x[2]) + 0.34837805869263727*$(x[3])) + -0.7242129606035013*softplus(0.17156635132643272 + 0.07330337401840348*$(x[1]) + -0.17735974291534173*$(x[2]) + -0.9111929848564051*$(x[3])) + 0.6225416211010737*softplus(0.016605037927807764 + 0.8199189360400561*$(x[1]) + 0.588507463268042*$(x[2]) + 0.6043543184008882*$(x[3])))) + softplus(-0.849383012756403 + 0.06473571727039973*softplus(-0.4631148608127349 + -0.7428552262850863*softplus(-0.5341243579019301 + 0.5611324607258696*$(x[1]) + -0.593816232159198*$(x[2]) + 0.34837805869263727*$(x[3])) + -0.05821446851395917*softplus(0.17156635132643272 + 0.07330337401840348*$(x[1]) + -0.17735974291534173*$(x[2]) + -0.9111929848564051*$(x[3])) + -0.035737768406546966*softplus(0.016605037927807764 + 0.8199189360400561*$(x[1]) + 0.588507463268042*$(x[2]) + 0.6043543184008882*$(x[3]))) + 0.959390902230969*softplus(0.5466854688811069 + 0.5410217402196289*softplus(-0.5341243579019301 + 0.5611324607258696*$(x[1]) + -0.593816232159198*$(x[2]) + 0.34837805869263727*$(x[3])) + -0.43803755847655035*softplus(0.17156635132643272 + 0.07330337401840348*$(x[1]) + -0.17735974291534173*$(x[2]) + -0.9111929848564051*$(x[3])) + 0.6527089787461371*softplus(0.016605037927807764 + 0.8199189360400561*$(x[1]) + 0.588507463268042*$(x[2]) + 0.6043543184008882*$(x[3]))) + 0.9101177155477291*softplus(0.6283785987613308 + 0.5367607999947079*softplus(-0.5341243579019301 + 0.5611324607258696*$(x[1]) + -0.593816232159198*$(x[2]) + 0.34837805869263727*$(x[3])) + -0.7242129606035013*softplus(0.17156635132643272 + 0.07330337401840348*$(x[1]) + -0.17735974291534173*$(x[2]) + -0.9111929848564051*$(x[3])) + 0.6225416211010737*softplus(0.016605037927807764 + 0.8199189360400561*$(x[1]) + 0.588507463268042*$(x[2]) + 0.6043543184008882*$(x[3])))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    