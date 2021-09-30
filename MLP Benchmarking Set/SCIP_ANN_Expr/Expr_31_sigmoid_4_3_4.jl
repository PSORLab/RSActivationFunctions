using JuMP, EAGO

                     m = Model()

                     register(m, :tsigmoid, 1, tsigmoid, autodiff = true)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -44.24614137611425 <= q <= 46.741438014815614)

                     add_NL_constraint(m, :(tsigmoid(0.6508401337587375 + 0.24881375368712755*tsigmoid(-0.7924137959953987 + -0.24606983294821072*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + -0.21648206565153982*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + 0.09641270088137643*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + 0.6678402753961832*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4]))) + -0.7173689375281502*tsigmoid(-0.801805933598724 + 0.970599707350909*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + 0.4101522067351495*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + 0.746328559792043*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + -0.6871332993965744*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4]))) + 0.36134276765783513*tsigmoid(0.7097356592005863 + -0.47686734397021047*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + -0.6302737874149136*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + -0.8260114349210217*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + -0.4262139142579233*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4]))) + -0.8487881606337573*tsigmoid(0.25287491718579247 + 0.21643752373772118*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + -0.6572655596584305*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + 0.45016768228767745*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + -0.9912523548080712*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4])))) + tsigmoid(0.0021486535726435996 + -0.35348197241690293*tsigmoid(-0.7924137959953987 + -0.24606983294821072*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + -0.21648206565153982*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + 0.09641270088137643*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + 0.6678402753961832*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4]))) + -0.6215297319483053*tsigmoid(-0.801805933598724 + 0.970599707350909*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + 0.4101522067351495*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + 0.746328559792043*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + -0.6871332993965744*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4]))) + -0.3566286436876078*tsigmoid(0.7097356592005863 + -0.47686734397021047*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + -0.6302737874149136*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + -0.8260114349210217*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + -0.4262139142579233*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4]))) + -0.5103356130788548*tsigmoid(0.25287491718579247 + 0.21643752373772118*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + -0.6572655596584305*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + 0.45016768228767745*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + -0.9912523548080712*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4])))) + tsigmoid(-0.24605551730029385 + 0.6912438859147505*tsigmoid(-0.7924137959953987 + -0.24606983294821072*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + -0.21648206565153982*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + 0.09641270088137643*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + 0.6678402753961832*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4]))) + 0.38549353533767006*tsigmoid(-0.801805933598724 + 0.970599707350909*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + 0.4101522067351495*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + 0.746328559792043*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + -0.6871332993965744*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4]))) + 0.7167681245025972*tsigmoid(0.7097356592005863 + -0.47686734397021047*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + -0.6302737874149136*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + -0.8260114349210217*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + -0.4262139142579233*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4]))) + 0.032577378897180154*tsigmoid(0.25287491718579247 + 0.21643752373772118*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + -0.6572655596584305*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + 0.45016768228767745*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + -0.9912523548080712*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4])))) + tsigmoid(-0.5055636446883458 + 0.6825990989165005*tsigmoid(-0.7924137959953987 + -0.24606983294821072*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + -0.21648206565153982*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + 0.09641270088137643*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + 0.6678402753961832*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4]))) + -0.03975683190600776*tsigmoid(-0.801805933598724 + 0.970599707350909*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + 0.4101522067351495*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + 0.746328559792043*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + -0.6871332993965744*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4]))) + -0.5255936262613066*tsigmoid(0.7097356592005863 + -0.47686734397021047*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + -0.6302737874149136*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + -0.8260114349210217*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + -0.4262139142579233*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4]))) + -0.7688251688231329*tsigmoid(0.25287491718579247 + 0.21643752373772118*tsigmoid(-0.42578272034045694 + -0.6930405389444534*$(x[1]) + 0.3035121372853422*$(x[2]) + -0.9670700177625537*$(x[3]) + 0.24675330974674692*$(x[4])) + -0.6572655596584305*tsigmoid(0.8302317037344693 + -0.6614237014658788*$(x[1]) + -0.9207656388740078*$(x[2]) + -0.9121646076935721*$(x[3]) + -0.465033979837993*$(x[4])) + 0.45016768228767745*tsigmoid(-0.11882955935960338 + 0.08624280374403304*$(x[1]) + 0.922777466622613*$(x[2]) + -0.04069033711623016*$(x[3]) + -0.7399089063380639*$(x[4])) + -0.9912523548080712*tsigmoid(-0.0687226878196352 + 0.21251973982781003*$(x[1]) + 0.538219101937286*$(x[2]) + -0.8538580105597169*$(x[3]) + -0.9016579579965711*$(x[4])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    