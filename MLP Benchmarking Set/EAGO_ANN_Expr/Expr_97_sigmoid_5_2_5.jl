using JuMP, EAGO

                     m = Model()

                     register(m, :tsigmoid, 1, tsigmoid, autodiff = true)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -31.30442997643422 <= q <= 34.2191768183764)

                     add_NL_constraint(m, :(tsigmoid(-0.9255358318781552 + -0.3327271241026599*tsigmoid(-0.09570590849812888 + 0.5592326731939425*$(x[1]) + 0.4107932100285794*$(x[2]) + 0.41425860300373873*$(x[3]) + 0.41487514232139633*$(x[4]) + 0.4529109908686526*$(x[5])) + 0.9441909082044369*tsigmoid(-0.3820590338927752 + 0.6942922976971757*$(x[1]) + -0.5084676517709501*$(x[2]) + -0.3374150070570874*$(x[3]) + 0.7022401286136413*$(x[4]) + -0.5556363818332013*$(x[5])) + -0.45597103520710736*tsigmoid(-0.7233009197288554 + -0.45749325220304504*$(x[1]) + -0.12410329342312654*$(x[2]) + 0.36733555033232834*$(x[3]) + -0.1538938609454581*$(x[4]) + 0.9667975217576799*$(x[5])) + 0.11237433761184956*tsigmoid(0.5166788934268638 + -0.23429490602165037*$(x[1]) + -0.8763841991017265*$(x[2]) + 0.6736077522242776*$(x[3]) + 0.32087301055872386*$(x[4]) + 0.8134982006493678*$(x[5])) + 0.6164496930520778*tsigmoid(-0.9026148193681314 + 0.14600782250524968*$(x[1]) + -0.3175794018135458*$(x[2]) + -0.259256983358088*$(x[3]) + -0.3749389241313388*$(x[4]) + 0.949539004938655*$(x[5]))) + tsigmoid(-0.11780692014046101 + 0.3046396327369587*tsigmoid(-0.09570590849812888 + 0.5592326731939425*$(x[1]) + 0.4107932100285794*$(x[2]) + 0.41425860300373873*$(x[3]) + 0.41487514232139633*$(x[4]) + 0.4529109908686526*$(x[5])) + 0.8574211763051354*tsigmoid(-0.3820590338927752 + 0.6942922976971757*$(x[1]) + -0.5084676517709501*$(x[2]) + -0.3374150070570874*$(x[3]) + 0.7022401286136413*$(x[4]) + -0.5556363818332013*$(x[5])) + -0.9101677750750041*tsigmoid(-0.7233009197288554 + -0.45749325220304504*$(x[1]) + -0.12410329342312654*$(x[2]) + 0.36733555033232834*$(x[3]) + -0.1538938609454581*$(x[4]) + 0.9667975217576799*$(x[5])) + 0.7571539834074605*tsigmoid(0.5166788934268638 + -0.23429490602165037*$(x[1]) + -0.8763841991017265*$(x[2]) + 0.6736077522242776*$(x[3]) + 0.32087301055872386*$(x[4]) + 0.8134982006493678*$(x[5])) + -0.4313060042730572*tsigmoid(-0.9026148193681314 + 0.14600782250524968*$(x[1]) + -0.3175794018135458*$(x[2]) + -0.259256983358088*$(x[3]) + -0.3749389241313388*$(x[4]) + 0.949539004938655*$(x[5]))) + tsigmoid(-0.7156917279256065 + 0.3311543458334656*tsigmoid(-0.09570590849812888 + 0.5592326731939425*$(x[1]) + 0.4107932100285794*$(x[2]) + 0.41425860300373873*$(x[3]) + 0.41487514232139633*$(x[4]) + 0.4529109908686526*$(x[5])) + 0.6183555215014183*tsigmoid(-0.3820590338927752 + 0.6942922976971757*$(x[1]) + -0.5084676517709501*$(x[2]) + -0.3374150070570874*$(x[3]) + 0.7022401286136413*$(x[4]) + -0.5556363818332013*$(x[5])) + -0.6332235171300846*tsigmoid(-0.7233009197288554 + -0.45749325220304504*$(x[1]) + -0.12410329342312654*$(x[2]) + 0.36733555033232834*$(x[3]) + -0.1538938609454581*$(x[4]) + 0.9667975217576799*$(x[5])) + 0.30430486649999233*tsigmoid(0.5166788934268638 + -0.23429490602165037*$(x[1]) + -0.8763841991017265*$(x[2]) + 0.6736077522242776*$(x[3]) + 0.32087301055872386*$(x[4]) + 0.8134982006493678*$(x[5])) + -0.20141112461243393*tsigmoid(-0.9026148193681314 + 0.14600782250524968*$(x[1]) + -0.3175794018135458*$(x[2]) + -0.259256983358088*$(x[3]) + -0.3749389241313388*$(x[4]) + 0.949539004938655*$(x[5]))) + tsigmoid(0.34424862539454937 + -0.6029087986372885*tsigmoid(-0.09570590849812888 + 0.5592326731939425*$(x[1]) + 0.4107932100285794*$(x[2]) + 0.41425860300373873*$(x[3]) + 0.41487514232139633*$(x[4]) + 0.4529109908686526*$(x[5])) + 0.1598093067579458*tsigmoid(-0.3820590338927752 + 0.6942922976971757*$(x[1]) + -0.5084676517709501*$(x[2]) + -0.3374150070570874*$(x[3]) + 0.7022401286136413*$(x[4]) + -0.5556363818332013*$(x[5])) + -0.6224720478724719*tsigmoid(-0.7233009197288554 + -0.45749325220304504*$(x[1]) + -0.12410329342312654*$(x[2]) + 0.36733555033232834*$(x[3]) + -0.1538938609454581*$(x[4]) + 0.9667975217576799*$(x[5])) + 0.2716613494346425*tsigmoid(0.5166788934268638 + -0.23429490602165037*$(x[1]) + -0.8763841991017265*$(x[2]) + 0.6736077522242776*$(x[3]) + 0.32087301055872386*$(x[4]) + 0.8134982006493678*$(x[5])) + -0.9268068406533909*tsigmoid(-0.9026148193681314 + 0.14600782250524968*$(x[1]) + -0.3175794018135458*$(x[2]) + -0.259256983358088*$(x[3]) + -0.3749389241313388*$(x[4]) + 0.949539004938655*$(x[5]))) + tsigmoid(-0.07506301402399762 + 0.48663885004037466*tsigmoid(-0.09570590849812888 + 0.5592326731939425*$(x[1]) + 0.4107932100285794*$(x[2]) + 0.41425860300373873*$(x[3]) + 0.41487514232139633*$(x[4]) + 0.4529109908686526*$(x[5])) + -0.8623718622464351*tsigmoid(-0.3820590338927752 + 0.6942922976971757*$(x[1]) + -0.5084676517709501*$(x[2]) + -0.3374150070570874*$(x[3]) + 0.7022401286136413*$(x[4]) + -0.5556363818332013*$(x[5])) + 0.8481164990938943*tsigmoid(-0.7233009197288554 + -0.45749325220304504*$(x[1]) + -0.12410329342312654*$(x[2]) + 0.36733555033232834*$(x[3]) + -0.1538938609454581*$(x[4]) + 0.9667975217576799*$(x[5])) + 0.5503263463755612*tsigmoid(0.5166788934268638 + -0.23429490602165037*$(x[1]) + -0.8763841991017265*$(x[2]) + 0.6736077522242776*$(x[3]) + 0.32087301055872386*$(x[4]) + 0.8134982006493678*$(x[5])) + -0.5050735345371598*tsigmoid(-0.9026148193681314 + 0.14600782250524968*$(x[1]) + -0.3175794018135458*$(x[2]) + -0.259256983358088*$(x[3]) + -0.3749389241313388*$(x[4]) + 0.949539004938655*$(x[5]))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    