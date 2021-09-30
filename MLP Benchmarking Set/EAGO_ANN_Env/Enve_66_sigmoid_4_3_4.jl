using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -45.85117670643175 <= q <= 44.25587583807973)

                     add_NL_constraint(m, :(sigmoid(-0.726655168682937 + 0.9743376079256607*sigmoid(0.14962324777026392 + -0.6792118446832189*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + -0.9933034898027775*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + -0.7390914843719725*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.28560159491909953*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4]))) + -0.5079893031311973*sigmoid(0.4374584746309784 + -0.7792155962273521*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + 0.9514248739449269*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + -0.15445514357978185*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.34180015324520907*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4]))) + -0.9740756718264025*sigmoid(-0.03961107816946763 + 0.9264202918501265*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + 0.9498141535178712*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + 0.5061920587277688*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.37967978354098264*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4]))) + 0.6083431608269221*sigmoid(0.2800719471596107 + 0.01291764248403604*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + -0.4465341242949652*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + -0.5188311123154565*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.24632117358990868*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4])))) + sigmoid(0.5667553270928747 + -0.9420061000185251*sigmoid(0.14962324777026392 + -0.6792118446832189*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + -0.9933034898027775*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + -0.7390914843719725*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.28560159491909953*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4]))) + 0.8883714633595599*sigmoid(0.4374584746309784 + -0.7792155962273521*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + 0.9514248739449269*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + -0.15445514357978185*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.34180015324520907*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4]))) + -0.5644638814858869*sigmoid(-0.03961107816946763 + 0.9264202918501265*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + 0.9498141535178712*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + 0.5061920587277688*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.37967978354098264*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4]))) + 0.7771264078746571*sigmoid(0.2800719471596107 + 0.01291764248403604*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + -0.4465341242949652*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + -0.5188311123154565*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.24632117358990868*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4])))) + sigmoid(0.8186369197348045 + -0.7252172022473045*sigmoid(0.14962324777026392 + -0.6792118446832189*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + -0.9933034898027775*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + -0.7390914843719725*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.28560159491909953*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4]))) + 0.194441621440955*sigmoid(0.4374584746309784 + -0.7792155962273521*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + 0.9514248739449269*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + -0.15445514357978185*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.34180015324520907*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4]))) + 0.4309802001890102*sigmoid(-0.03961107816946763 + 0.9264202918501265*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + 0.9498141535178712*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + 0.5061920587277688*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.37967978354098264*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4]))) + 0.28976654390875645*sigmoid(0.2800719471596107 + 0.01291764248403604*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + -0.4465341242949652*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + -0.5188311123154565*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.24632117358990868*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4])))) + sigmoid(0.9761154143554873 + 0.3684488633235983*sigmoid(0.14962324777026392 + -0.6792118446832189*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + -0.9933034898027775*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + -0.7390914843719725*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.28560159491909953*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4]))) + 0.6041314602687438*sigmoid(0.4374584746309784 + -0.7792155962273521*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + 0.9514248739449269*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + -0.15445514357978185*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.34180015324520907*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4]))) + -0.8659083014846818*sigmoid(-0.03961107816946763 + 0.9264202918501265*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + 0.9498141535178712*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + 0.5061920587277688*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.37967978354098264*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4]))) + -0.5685608762623033*sigmoid(0.2800719471596107 + 0.01291764248403604*sigmoid(0.9780562405839417 + -0.25675114562624124*$(x[1]) + 0.21245931981101407*$(x[2]) + -0.8811804177299982*$(x[3]) + 0.9662257346376735*$(x[4])) + -0.4465341242949652*sigmoid(-0.9461526940667064 + 0.2372664711565089*$(x[1]) + 0.04798436581096821*$(x[2]) + -0.7946809788675315*$(x[3]) + 0.6553830565360919*$(x[4])) + -0.5188311123154565*sigmoid(-0.8906891799402641 + -0.9685132036735742*$(x[1]) + 0.7077146576189697*$(x[2]) + -0.426363099291998*$(x[3]) + -0.758503030072923*$(x[4])) + -0.24632117358990868*sigmoid(-0.513746243745993 + -0.8318873143899537*$(x[1]) + 0.05136832081782039*$(x[2]) + 0.6705320116477487*$(x[3]) + -0.5851563608791701*$(x[4])))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    