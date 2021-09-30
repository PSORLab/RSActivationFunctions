using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -19.699408402298598 <= q <= 22.796397936605)

                     add_NL_constraint(m, :(sigmoid(0.09867518995885849 + -0.4039145072561916*sigmoid(-0.7984999946586226 + 0.5454698196401755*$(x[1]) + 0.002029564318569932*$(x[2]) + -0.9703445000753295*$(x[3]) + -0.23503216875830013*$(x[4]) + -0.654879965800296*$(x[5])) + 0.8322880851465859*sigmoid(-0.32940250961246464 + -0.9592046612776461*$(x[1]) + -0.2913109658608066*$(x[2]) + 0.6555112929401892*$(x[3]) + -0.26851235746884283*$(x[4]) + -0.06441169476048891*$(x[5])) + 0.2374912283224373*sigmoid(-0.22647534024737315 + -0.9301285806587103*$(x[1]) + 0.10713249219622512*$(x[2]) + -0.7698145583277172*$(x[3]) + -0.29378966758715963*$(x[4]) + -0.6064408965403327*$(x[5])) + 0.6269672666392174*sigmoid(-0.6841603959155296 + -0.32557724721928594*$(x[1]) + -0.7182409525528026*$(x[2]) + 0.8201972763678005*$(x[3]) + -0.03851935809469298*$(x[4]) + -0.054076879231296004*$(x[5]))) + sigmoid(0.839724767319808 + -0.7464971507157427*sigmoid(-0.7984999946586226 + 0.5454698196401755*$(x[1]) + 0.002029564318569932*$(x[2]) + -0.9703445000753295*$(x[3]) + -0.23503216875830013*$(x[4]) + -0.654879965800296*$(x[5])) + -0.9757317628644415*sigmoid(-0.32940250961246464 + -0.9592046612776461*$(x[1]) + -0.2913109658608066*$(x[2]) + 0.6555112929401892*$(x[3]) + -0.26851235746884283*$(x[4]) + -0.06441169476048891*$(x[5])) + -0.8023493319612838*sigmoid(-0.22647534024737315 + -0.9301285806587103*$(x[1]) + 0.10713249219622512*$(x[2]) + -0.7698145583277172*$(x[3]) + -0.29378966758715963*$(x[4]) + -0.6064408965403327*$(x[5])) + -0.8082614338071004*sigmoid(-0.6841603959155296 + -0.32557724721928594*$(x[1]) + -0.7182409525528026*$(x[2]) + 0.8201972763678005*$(x[3]) + -0.03851935809469298*$(x[4]) + -0.054076879231296004*$(x[5]))) + sigmoid(-0.021068371466398172 + -0.06936467353416642*sigmoid(-0.7984999946586226 + 0.5454698196401755*$(x[1]) + 0.002029564318569932*$(x[2]) + -0.9703445000753295*$(x[3]) + -0.23503216875830013*$(x[4]) + -0.654879965800296*$(x[5])) + 0.4222466665565201*sigmoid(-0.32940250961246464 + -0.9592046612776461*$(x[1]) + -0.2913109658608066*$(x[2]) + 0.6555112929401892*$(x[3]) + -0.26851235746884283*$(x[4]) + -0.06441169476048891*$(x[5])) + -0.9415388631995376*sigmoid(-0.22647534024737315 + -0.9301285806587103*$(x[1]) + 0.10713249219622512*$(x[2]) + -0.7698145583277172*$(x[3]) + -0.29378966758715963*$(x[4]) + -0.6064408965403327*$(x[5])) + 0.2611835591472529*sigmoid(-0.6841603959155296 + -0.32557724721928594*$(x[1]) + -0.7182409525528026*$(x[2]) + 0.8201972763678005*$(x[3]) + -0.03851935809469298*$(x[4]) + -0.054076879231296004*$(x[5]))) + sigmoid(-0.030871335971588643 + 0.18036889076853857*sigmoid(-0.7984999946586226 + 0.5454698196401755*$(x[1]) + 0.002029564318569932*$(x[2]) + -0.9703445000753295*$(x[3]) + -0.23503216875830013*$(x[4]) + -0.654879965800296*$(x[5])) + -0.9701318678691977*sigmoid(-0.32940250961246464 + -0.9592046612776461*$(x[1]) + -0.2913109658608066*$(x[2]) + 0.6555112929401892*$(x[3]) + -0.26851235746884283*$(x[4]) + -0.06441169476048891*$(x[5])) + 0.03730863567389875*sigmoid(-0.22647534024737315 + -0.9301285806587103*$(x[1]) + 0.10713249219622512*$(x[2]) + -0.7698145583277172*$(x[3]) + -0.29378966758715963*$(x[4]) + -0.6064408965403327*$(x[5])) + 0.984729475594829*sigmoid(-0.6841603959155296 + -0.32557724721928594*$(x[1]) + -0.7182409525528026*$(x[2]) + 0.8201972763678005*$(x[3]) + -0.03851935809469298*$(x[4]) + -0.054076879231296004*$(x[5]))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    