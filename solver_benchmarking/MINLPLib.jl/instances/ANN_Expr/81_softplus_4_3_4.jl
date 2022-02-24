using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -32.35079088597315 <= q <= 33.04707617725232)

                     add_NL_constraint(m, :(log(1 + exp(0.7516814527569986 + 0.07211062597860263*log(1 + exp(0.800956367833014 + 0.1575422515944389*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + 0.30181017969165413*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + 0.5024447348308962*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + -0.7463877606567912*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))) + -0.28099609060437203*log(1 + exp(-0.8241959750660866 + 0.48752067922981546*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + -0.8942226763869914*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + -0.1882956347370066*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + 0.0613423792438379*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))) + 0.19401527858092393*log(1 + exp(-0.014178711859067938 + -0.6496362145686132*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + 0.5744456236018198*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + 0.06990002879133872*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + 0.11738131892224501*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))) + -0.43991506890783594*log(1 + exp(-0.38559964579408224 + -0.6882459841770032*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + 0.5869261088566509*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + 0.7286086900917508*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + 0.9385542676650069*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))))) + log(1 + exp(0.5442086655639207 + -0.859033717055464*log(1 + exp(0.800956367833014 + 0.1575422515944389*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + 0.30181017969165413*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + 0.5024447348308962*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + -0.7463877606567912*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))) + -0.6994107397450926*log(1 + exp(-0.8241959750660866 + 0.48752067922981546*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + -0.8942226763869914*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + -0.1882956347370066*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + 0.0613423792438379*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))) + -0.21447765318835765*log(1 + exp(-0.014178711859067938 + -0.6496362145686132*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + 0.5744456236018198*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + 0.06990002879133872*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + 0.11738131892224501*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))) + 0.9247182813449877*log(1 + exp(-0.38559964579408224 + -0.6882459841770032*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + 0.5869261088566509*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + 0.7286086900917508*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + 0.9385542676650069*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))))) + log(1 + exp(-0.33457802196328057 + 0.8511359109817849*log(1 + exp(0.800956367833014 + 0.1575422515944389*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + 0.30181017969165413*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + 0.5024447348308962*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + -0.7463877606567912*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))) + 0.9056512987027685*log(1 + exp(-0.8241959750660866 + 0.48752067922981546*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + -0.8942226763869914*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + -0.1882956347370066*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + 0.0613423792438379*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))) + -0.901983268277478*log(1 + exp(-0.014178711859067938 + -0.6496362145686132*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + 0.5744456236018198*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + 0.06990002879133872*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + 0.11738131892224501*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))) + -0.02854306019224717*log(1 + exp(-0.38559964579408224 + -0.6882459841770032*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + 0.5869261088566509*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + 0.7286086900917508*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + 0.9385542676650069*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))))) + log(1 + exp(0.85223750045835 + -0.7468516096711739*log(1 + exp(0.800956367833014 + 0.1575422515944389*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + 0.30181017969165413*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + 0.5024447348308962*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + -0.7463877606567912*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))) + -0.1269236474608202*log(1 + exp(-0.8241959750660866 + 0.48752067922981546*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + -0.8942226763869914*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + -0.1882956347370066*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + 0.0613423792438379*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))) + -0.5407492362268407*log(1 + exp(-0.014178711859067938 + -0.6496362145686132*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + 0.5744456236018198*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + 0.06990002879133872*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + 0.11738131892224501*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))) + -0.8670343803826994*log(1 + exp(-0.38559964579408224 + -0.6882459841770032*log(1 + exp(-0.7991039641167363 + 0.28486813994855353*$(x[1]) + 0.682420711072186*$(x[2]) + 0.05952972537514656*$(x[3]) + 0.998655851604032*$(x[4]))) + 0.5869261088566509*log(1 + exp(-0.9827986828893067 + 0.8570099551669048*$(x[1]) + -0.029287184975956393*$(x[2]) + 0.28818320654817864*$(x[3]) + -0.9376552616355038*$(x[4]))) + 0.7286086900917508*log(1 + exp(0.33306588949676286 + 0.667638331016386*$(x[1]) + 0.7691056200843742*$(x[2]) + 0.5908191290581821*$(x[3]) + 0.06769592108634237*$(x[4]))) + 0.9385542676650069*log(1 + exp(-0.7350489056562997 + 0.545921427040915*$(x[1]) + -0.3116746162014481*$(x[2]) + 0.844944868364411*$(x[3]) + -0.9086564944285445*$(x[4]))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    