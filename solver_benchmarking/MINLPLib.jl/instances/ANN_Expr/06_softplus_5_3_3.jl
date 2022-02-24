using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -6.148474362391325 <= q <= 10.677081718106185)

                     add_NL_constraint(m, :(log(1 + exp(-0.2518902526786948 + 0.9847866884384731*log(1 + exp(0.2752793536861313 + -0.1568397657479923*log(1 + exp(0.860587138591963 + -0.8175452953351838*$(x[1]) + -0.2911974841119127*$(x[2]) + -0.03381280038289569*$(x[3]) + -0.021805451939043596*$(x[4]) + 0.5056820468199099*$(x[5]))) + -0.1062303426643516*log(1 + exp(0.40215662414311426 + 0.47796901525165936*$(x[1]) + -0.8905174133817706*$(x[2]) + 0.3815447257070974*$(x[3]) + -0.7235559998197827*$(x[4]) + -0.43821768660130234*$(x[5]))) + 0.5642457669318222*log(1 + exp(-0.371366893645773 + 0.03574445051584929*$(x[1]) + 0.07916257211879385*$(x[2]) + 0.5096612381208421*$(x[3]) + 0.1651890232922466*$(x[4]) + -0.4691295251335359*$(x[5]))))) + -0.7649756572019379*log(1 + exp(0.9918790053549298 + 0.85981905523253*log(1 + exp(0.860587138591963 + -0.8175452953351838*$(x[1]) + -0.2911974841119127*$(x[2]) + -0.03381280038289569*$(x[3]) + -0.021805451939043596*$(x[4]) + 0.5056820468199099*$(x[5]))) + 0.9639515148457134*log(1 + exp(0.40215662414311426 + 0.47796901525165936*$(x[1]) + -0.8905174133817706*$(x[2]) + 0.3815447257070974*$(x[3]) + -0.7235559998197827*$(x[4]) + -0.43821768660130234*$(x[5]))) + -0.08816238542180388*log(1 + exp(-0.371366893645773 + 0.03574445051584929*$(x[1]) + 0.07916257211879385*$(x[2]) + 0.5096612381208421*$(x[3]) + 0.1651890232922466*$(x[4]) + -0.4691295251335359*$(x[5]))))) + 0.10087195663512549*log(1 + exp(-0.8202416682813327 + -0.17766510212211006*log(1 + exp(0.860587138591963 + -0.8175452953351838*$(x[1]) + -0.2911974841119127*$(x[2]) + -0.03381280038289569*$(x[3]) + -0.021805451939043596*$(x[4]) + 0.5056820468199099*$(x[5]))) + -0.1793020696087071*log(1 + exp(0.40215662414311426 + 0.47796901525165936*$(x[1]) + -0.8905174133817706*$(x[2]) + 0.3815447257070974*$(x[3]) + -0.7235559998197827*$(x[4]) + -0.43821768660130234*$(x[5]))) + -0.41892665263312834*log(1 + exp(-0.371366893645773 + 0.03574445051584929*$(x[1]) + 0.07916257211879385*$(x[2]) + 0.5096612381208421*$(x[3]) + 0.1651890232922466*$(x[4]) + -0.4691295251335359*$(x[5]))))))) + log(1 + exp(0.638435553115452 + 0.09453389081965424*log(1 + exp(0.2752793536861313 + -0.1568397657479923*log(1 + exp(0.860587138591963 + -0.8175452953351838*$(x[1]) + -0.2911974841119127*$(x[2]) + -0.03381280038289569*$(x[3]) + -0.021805451939043596*$(x[4]) + 0.5056820468199099*$(x[5]))) + -0.1062303426643516*log(1 + exp(0.40215662414311426 + 0.47796901525165936*$(x[1]) + -0.8905174133817706*$(x[2]) + 0.3815447257070974*$(x[3]) + -0.7235559998197827*$(x[4]) + -0.43821768660130234*$(x[5]))) + 0.5642457669318222*log(1 + exp(-0.371366893645773 + 0.03574445051584929*$(x[1]) + 0.07916257211879385*$(x[2]) + 0.5096612381208421*$(x[3]) + 0.1651890232922466*$(x[4]) + -0.4691295251335359*$(x[5]))))) + -0.07927014075141203*log(1 + exp(0.9918790053549298 + 0.85981905523253*log(1 + exp(0.860587138591963 + -0.8175452953351838*$(x[1]) + -0.2911974841119127*$(x[2]) + -0.03381280038289569*$(x[3]) + -0.021805451939043596*$(x[4]) + 0.5056820468199099*$(x[5]))) + 0.9639515148457134*log(1 + exp(0.40215662414311426 + 0.47796901525165936*$(x[1]) + -0.8905174133817706*$(x[2]) + 0.3815447257070974*$(x[3]) + -0.7235559998197827*$(x[4]) + -0.43821768660130234*$(x[5]))) + -0.08816238542180388*log(1 + exp(-0.371366893645773 + 0.03574445051584929*$(x[1]) + 0.07916257211879385*$(x[2]) + 0.5096612381208421*$(x[3]) + 0.1651890232922466*$(x[4]) + -0.4691295251335359*$(x[5]))))) + 0.2624391124261729*log(1 + exp(-0.8202416682813327 + -0.17766510212211006*log(1 + exp(0.860587138591963 + -0.8175452953351838*$(x[1]) + -0.2911974841119127*$(x[2]) + -0.03381280038289569*$(x[3]) + -0.021805451939043596*$(x[4]) + 0.5056820468199099*$(x[5]))) + -0.1793020696087071*log(1 + exp(0.40215662414311426 + 0.47796901525165936*$(x[1]) + -0.8905174133817706*$(x[2]) + 0.3815447257070974*$(x[3]) + -0.7235559998197827*$(x[4]) + -0.43821768660130234*$(x[5]))) + -0.41892665263312834*log(1 + exp(-0.371366893645773 + 0.03574445051584929*$(x[1]) + 0.07916257211879385*$(x[2]) + 0.5096612381208421*$(x[3]) + 0.1651890232922466*$(x[4]) + -0.4691295251335359*$(x[5]))))))) + log(1 + exp(0.5273095272255199 + 0.4025366346817978*log(1 + exp(0.2752793536861313 + -0.1568397657479923*log(1 + exp(0.860587138591963 + -0.8175452953351838*$(x[1]) + -0.2911974841119127*$(x[2]) + -0.03381280038289569*$(x[3]) + -0.021805451939043596*$(x[4]) + 0.5056820468199099*$(x[5]))) + -0.1062303426643516*log(1 + exp(0.40215662414311426 + 0.47796901525165936*$(x[1]) + -0.8905174133817706*$(x[2]) + 0.3815447257070974*$(x[3]) + -0.7235559998197827*$(x[4]) + -0.43821768660130234*$(x[5]))) + 0.5642457669318222*log(1 + exp(-0.371366893645773 + 0.03574445051584929*$(x[1]) + 0.07916257211879385*$(x[2]) + 0.5096612381208421*$(x[3]) + 0.1651890232922466*$(x[4]) + -0.4691295251335359*$(x[5]))))) + -0.585290488444234*log(1 + exp(0.9918790053549298 + 0.85981905523253*log(1 + exp(0.860587138591963 + -0.8175452953351838*$(x[1]) + -0.2911974841119127*$(x[2]) + -0.03381280038289569*$(x[3]) + -0.021805451939043596*$(x[4]) + 0.5056820468199099*$(x[5]))) + 0.9639515148457134*log(1 + exp(0.40215662414311426 + 0.47796901525165936*$(x[1]) + -0.8905174133817706*$(x[2]) + 0.3815447257070974*$(x[3]) + -0.7235559998197827*$(x[4]) + -0.43821768660130234*$(x[5]))) + -0.08816238542180388*log(1 + exp(-0.371366893645773 + 0.03574445051584929*$(x[1]) + 0.07916257211879385*$(x[2]) + 0.5096612381208421*$(x[3]) + 0.1651890232922466*$(x[4]) + -0.4691295251335359*$(x[5]))))) + 0.1647823489958915*log(1 + exp(-0.8202416682813327 + -0.17766510212211006*log(1 + exp(0.860587138591963 + -0.8175452953351838*$(x[1]) + -0.2911974841119127*$(x[2]) + -0.03381280038289569*$(x[3]) + -0.021805451939043596*$(x[4]) + 0.5056820468199099*$(x[5]))) + -0.1793020696087071*log(1 + exp(0.40215662414311426 + 0.47796901525165936*$(x[1]) + -0.8905174133817706*$(x[2]) + 0.3815447257070974*$(x[3]) + -0.7235559998197827*$(x[4]) + -0.43821768660130234*$(x[5]))) + -0.41892665263312834*log(1 + exp(-0.371366893645773 + 0.03574445051584929*$(x[1]) + 0.07916257211879385*$(x[2]) + 0.5096612381208421*$(x[3]) + 0.1651890232922466*$(x[4]) + -0.4691295251335359*$(x[5]))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    