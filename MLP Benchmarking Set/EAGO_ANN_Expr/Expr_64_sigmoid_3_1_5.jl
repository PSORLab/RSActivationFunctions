using JuMP, EAGO

                     m = Model()

                     register(m, :tsigmoid, 1, tsigmoid, autodiff = true)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -8.843258968176094 <= q <= 7.742439843954944)

                     add_NL_constraint(m, :(tsigmoid(0.708040130003313 + 0.42689932475287673*$(x[1]) + -0.8622598504299739*$(x[2]) + -0.9085925823905816*$(x[3])) + tsigmoid(-0.9869879186174217 + -0.8142422900494024*$(x[1]) + 0.571115023141878*$(x[2]) + 0.3481217041223639*$(x[3])) + tsigmoid(-0.013673110680600509 + -0.699424761625381*$(x[1]) + 0.2900859361857093*$(x[2]) + 0.8479582814657554*$(x[3])) + tsigmoid(-0.3900421921131745 + 0.28237850992924773*$(x[1]) + 0.20775012782016322*$(x[2]) + -0.5569936372597266*$(x[3])) + tsigmoid(0.1322535292973086 + 0.7398321839683919*$(x[1]) + 0.7173251709629187*$(x[2]) + 0.019870021961148066*$(x[3])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    