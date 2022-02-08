using JuMP, EAGO

                     m = Model()

                     register(m, :tswish, 1, tswish, autodiff = true)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -46.7612021435079 <= q <= 49.229147823929026)

                     add_NL_constraint(m, :(tswish(-0.27677481180629515 + 0.027283547602490632*tswish(0.8306239518544105 + -0.9810852963848697*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + -0.7626911459129988*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.594916524711997*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + -0.5957878874618423*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3]))) + -0.40896722445057154*tswish(0.7355348610133143 + -0.30147383434412367*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + 0.35443136329128233*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.015401527145070038*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + 0.15627085257914652*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3]))) + 0.5286228765642149*tswish(0.963428592183341 + -0.8091780219876465*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + 0.9927009026736471*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.8330678136226566*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + 0.5720256982756813*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3]))) + -0.9455620381717704*tswish(-0.9188837758426991 + 0.8104562733292617*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + -0.8492508519599911*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.8340928921648612*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + -0.8857399386529519*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3])))) + tswish(-0.639931042705554 + 0.4646012954401608*tswish(0.8306239518544105 + -0.9810852963848697*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + -0.7626911459129988*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.594916524711997*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + -0.5957878874618423*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3]))) + 0.8546650303812995*tswish(0.7355348610133143 + -0.30147383434412367*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + 0.35443136329128233*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.015401527145070038*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + 0.15627085257914652*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3]))) + 0.03265312702176226*tswish(0.963428592183341 + -0.8091780219876465*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + 0.9927009026736471*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.8330678136226566*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + 0.5720256982756813*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3]))) + -0.8773352958340506*tswish(-0.9188837758426991 + 0.8104562733292617*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + -0.8492508519599911*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.8340928921648612*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + -0.8857399386529519*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3])))) + tswish(0.10819873545085867 + 0.5093138203313128*tswish(0.8306239518544105 + -0.9810852963848697*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + -0.7626911459129988*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.594916524711997*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + -0.5957878874618423*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3]))) + 0.17418542942280624*tswish(0.7355348610133143 + -0.30147383434412367*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + 0.35443136329128233*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.015401527145070038*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + 0.15627085257914652*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3]))) + -0.3589056296025168*tswish(0.963428592183341 + -0.8091780219876465*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + 0.9927009026736471*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.8330678136226566*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + 0.5720256982756813*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3]))) + 0.7568261069876656*tswish(-0.9188837758426991 + 0.8104562733292617*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + -0.8492508519599911*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.8340928921648612*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + -0.8857399386529519*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3])))) + tswish(0.32361925631331045 + 0.07410559334069378*tswish(0.8306239518544105 + -0.9810852963848697*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + -0.7626911459129988*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.594916524711997*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + -0.5957878874618423*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3]))) + -0.9634177924933005*tswish(0.7355348610133143 + -0.30147383434412367*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + 0.35443136329128233*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.015401527145070038*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + 0.15627085257914652*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3]))) + -0.4000580358794781*tswish(0.963428592183341 + -0.8091780219876465*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + 0.9927009026736471*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.8330678136226566*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + 0.5720256982756813*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3]))) + 0.19827635012393197*tswish(-0.9188837758426991 + 0.8104562733292617*tswish(-0.9233382174555076 + -0.7574549656532357*$(x[1]) + 0.4193796034060262*$(x[2]) + -0.9589160913361483*$(x[3])) + -0.8492508519599911*tswish(-0.11139259062844564 + 0.29952735826249777*$(x[1]) + 0.8896690826002636*$(x[2]) + -0.07914206343658492*$(x[3])) + 0.8340928921648612*tswish(-0.7433287192318199 + 0.5307065467567265*$(x[1]) + 0.8053400818009657*$(x[2]) + -0.5286150053499892*$(x[3])) + -0.8857399386529519*tswish(-0.0535416501601409 + 0.8662358618697215*$(x[1]) + -0.5908223046502772*$(x[2]) + -0.8510877327366346*$(x[3])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    