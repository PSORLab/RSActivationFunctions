using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -84.18507402401383 <= q <= 74.38673610467701)

                     add_NL_constraint(m, :(swish(-0.8549087705894971 + 0.9543655440144265*swish(0.0751720465901844 + 0.879988538731995*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.526166288188262*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.9394368235594466*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.20512402492652626*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3])))) + -0.7100753503535358*swish(0.5060029042899505 + 0.5357082869854093*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.026979859210937818*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + -0.099873457609573*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + -0.19909406173404998*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3])))) + 0.21077773769339014*swish(-0.6168321301535191 + 0.08078815745338863*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + -0.896878690062942*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.2758129774152658*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.06441990710306333*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3])))) + -0.541364723515501*swish(0.6501694598335028 + 0.1240474544060235*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.9817318317259809*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.11391063175544547*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.14331149010450783*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))))) + swish(-0.6986665227225246 + -0.43075659990375614*swish(0.0751720465901844 + 0.879988538731995*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.526166288188262*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.9394368235594466*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.20512402492652626*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3])))) + -0.048974022225801495*swish(0.5060029042899505 + 0.5357082869854093*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.026979859210937818*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + -0.099873457609573*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + -0.19909406173404998*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3])))) + 0.40914129449637615*swish(-0.6168321301535191 + 0.08078815745338863*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + -0.896878690062942*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.2758129774152658*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.06441990710306333*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3])))) + -0.8190449815551784*swish(0.6501694598335028 + 0.1240474544060235*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.9817318317259809*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.11391063175544547*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.14331149010450783*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))))) + swish(-0.07497835683704457 + 0.3830050524451014*swish(0.0751720465901844 + 0.879988538731995*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.526166288188262*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.9394368235594466*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.20512402492652626*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3])))) + -0.24285847733651433*swish(0.5060029042899505 + 0.5357082869854093*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.026979859210937818*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + -0.099873457609573*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + -0.19909406173404998*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3])))) + 0.24370645479170738*swish(-0.6168321301535191 + 0.08078815745338863*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + -0.896878690062942*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.2758129774152658*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.06441990710306333*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3])))) + -0.05409327380738471*swish(0.6501694598335028 + 0.1240474544060235*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.9817318317259809*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.11391063175544547*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.14331149010450783*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))))) + swish(-0.5862561427459396 + 0.5621771558595174*swish(0.0751720465901844 + 0.879988538731995*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.526166288188262*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.9394368235594466*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.20512402492652626*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3])))) + -0.3091992460181925*swish(0.5060029042899505 + 0.5357082869854093*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.026979859210937818*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + -0.099873457609573*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + -0.19909406173404998*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3])))) + 0.4274338579310326*swish(-0.6168321301535191 + 0.08078815745338863*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + -0.896878690062942*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.2758129774152658*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.06441990710306333*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3])))) + -0.7338970609011959*swish(0.6501694598335028 + 0.1240474544060235*swish(0.2410920244187622 + -0.8549165791409097*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.9042335680991518*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.033073480690713364*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.46133331353770357*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.9817318317259809*swish(0.7077398014210163 + -0.8875525102289932*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.3114800182616828*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.6793396082347858*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.8286351134939691*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.11391063175544547*swish(0.08967757530546994 + -0.6772577714184207*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + 0.0862261456529323*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + 0.8438271826503776*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + 0.4691446749630801*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))) + 0.14331149010450783*swish(-0.6509478374906199 + -0.2869456619600026*swish(0.8498846297184706 + -0.00048046832249104554*$(x[1]) + -0.8943335083635202*$(x[2]) + -0.7908783617038644*$(x[3])) + -0.6189819073825915*swish(-0.3004127798937475 + 0.9774986116980324*$(x[1]) + -0.32529835964385123*$(x[2]) + 0.31138306333118315*$(x[3])) + -0.8919061378952815*swish(-0.996856280642318 + 0.8143558542644809*$(x[1]) + 0.47005874333527986*$(x[2]) + 0.0020119255234156697*$(x[3])) + -0.28490486648765456*swish(0.8600902168446733 + 0.9827501667990508*$(x[1]) + 0.8665317243359536*$(x[2]) + -0.3905246461058671*$(x[3]))))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    