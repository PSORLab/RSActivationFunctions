using JuMP, EAGO

                     m = Model()

                     register(m, :tgelu, 1, tgelu, autodiff = true)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -25.033593843816078 <= q <= 30.771943318144178)

                     add_NL_constraint(m, :(tgelu(-0.21043262598581025 + -0.43985914532472625*tgelu(0.741143162034049 + -0.0759123677117417*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.647222358343789*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.4654166346308939*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.17817483392419575*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.8877106383986808*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + 0.9895861040156801*tgelu(-0.7835263712809195 + 0.42359752148741947*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.5492426609830434*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.020993142290360645*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + -0.5046481405608971*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.19256631925466472*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + -0.9526742144101048*tgelu(0.5167893899113523 + 0.9088383543538403*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.8777251439462628*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.02013732685611025*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + -0.6229176762232775*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + 0.6176416598343679*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + 0.4546001813699596*tgelu(-0.2693163055042085 + -0.10811873801263383*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.17159767188449493*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + 0.5006708213421951*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.053669241529314515*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.6242179589162236*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + 0.37602793713881644*tgelu(0.28814624669718114 + -0.003304159090915082*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.18979043807578977*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.5751737799195085*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.12452886410581554*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.3735663561956235*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4])))) + tgelu(-0.5961661610225923 + 0.5548623932685306*tgelu(0.741143162034049 + -0.0759123677117417*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.647222358343789*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.4654166346308939*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.17817483392419575*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.8877106383986808*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + 0.46750667212208397*tgelu(-0.7835263712809195 + 0.42359752148741947*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.5492426609830434*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.020993142290360645*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + -0.5046481405608971*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.19256631925466472*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + -0.3136278056221373*tgelu(0.5167893899113523 + 0.9088383543538403*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.8777251439462628*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.02013732685611025*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + -0.6229176762232775*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + 0.6176416598343679*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + -0.6484765948815219*tgelu(-0.2693163055042085 + -0.10811873801263383*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.17159767188449493*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + 0.5006708213421951*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.053669241529314515*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.6242179589162236*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + -0.04251065840650403*tgelu(0.28814624669718114 + -0.003304159090915082*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.18979043807578977*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.5751737799195085*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.12452886410581554*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.3735663561956235*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4])))) + tgelu(-0.9287335767654556 + 0.20951817438107279*tgelu(0.741143162034049 + -0.0759123677117417*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.647222358343789*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.4654166346308939*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.17817483392419575*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.8877106383986808*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + 0.24776672163556546*tgelu(-0.7835263712809195 + 0.42359752148741947*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.5492426609830434*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.020993142290360645*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + -0.5046481405608971*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.19256631925466472*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + 0.042715007968098906*tgelu(0.5167893899113523 + 0.9088383543538403*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.8777251439462628*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.02013732685611025*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + -0.6229176762232775*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + 0.6176416598343679*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + -0.0767361806663196*tgelu(-0.2693163055042085 + -0.10811873801263383*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.17159767188449493*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + 0.5006708213421951*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.053669241529314515*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.6242179589162236*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + -0.2566644489308709*tgelu(0.28814624669718114 + -0.003304159090915082*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.18979043807578977*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.5751737799195085*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.12452886410581554*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.3735663561956235*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4])))) + tgelu(0.20706172245591192 + -0.5568716109305356*tgelu(0.741143162034049 + -0.0759123677117417*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.647222358343789*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.4654166346308939*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.17817483392419575*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.8877106383986808*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + -0.3346669186360902*tgelu(-0.7835263712809195 + 0.42359752148741947*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.5492426609830434*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.020993142290360645*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + -0.5046481405608971*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.19256631925466472*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + -0.9376191464125134*tgelu(0.5167893899113523 + 0.9088383543538403*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.8777251439462628*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.02013732685611025*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + -0.6229176762232775*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + 0.6176416598343679*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + 0.3645783555383342*tgelu(-0.2693163055042085 + -0.10811873801263383*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.17159767188449493*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + 0.5006708213421951*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.053669241529314515*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.6242179589162236*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + -0.26705521696668555*tgelu(0.28814624669718114 + -0.003304159090915082*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.18979043807578977*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.5751737799195085*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.12452886410581554*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.3735663561956235*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4])))) + tgelu(0.7173816125628276 + -0.06679756778660373*tgelu(0.741143162034049 + -0.0759123677117417*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.647222358343789*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.4654166346308939*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.17817483392419575*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.8877106383986808*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + -0.4903613362313304*tgelu(-0.7835263712809195 + 0.42359752148741947*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.5492426609830434*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.020993142290360645*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + -0.5046481405608971*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.19256631925466472*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + 0.33971373843491026*tgelu(0.5167893899113523 + 0.9088383543538403*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.8777251439462628*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.02013732685611025*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + -0.6229176762232775*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + 0.6176416598343679*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + 0.41174115257663635*tgelu(-0.2693163055042085 + -0.10811873801263383*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.17159767188449493*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + 0.5006708213421951*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.053669241529314515*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.6242179589162236*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4]))) + -0.42114649148906613*tgelu(0.28814624669718114 + -0.003304159090915082*tgelu(-0.6355272657894862 + -0.43274405797692816*$(x[1]) + -0.44116171545097504*$(x[2]) + 0.05521658055624057*$(x[3]) + 0.26660076016603096*$(x[4])) + -0.18979043807578977*tgelu(-0.32351494105166534 + -0.14455050762895372*$(x[1]) + -0.40164699564090256*$(x[2]) + -0.6718728787463042*$(x[3]) + 0.19323943152196232*$(x[4])) + -0.5751737799195085*tgelu(0.22617857152669574 + 0.6148198775491975*$(x[1]) + -0.7365432208361611*$(x[2]) + 0.5265400297798424*$(x[3]) + 0.1720946950229485*$(x[4])) + 0.12452886410581554*tgelu(0.6455611369337797 + 0.44581953253876705*$(x[1]) + 0.3181475235671867*$(x[2]) + -0.6989552718816459*$(x[3]) + -0.9065267584864745*$(x[4])) + -0.3735663561956235*tgelu(-0.6186793134964015 + -0.20504362817722255*$(x[1]) + 0.03361995913991844*$(x[2]) + 0.26181622877396027*$(x[3]) + 0.5771640661747139*$(x[4])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    