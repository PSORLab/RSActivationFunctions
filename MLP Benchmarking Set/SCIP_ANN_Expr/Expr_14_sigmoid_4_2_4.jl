using JuMP, EAGO

                     m = Model()

                     register(m, :tsigmoid, 1, tsigmoid, autodiff = true)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -18.018920640159507 <= q <= 15.482184217316266)

                     add_NL_constraint(m, :(tsigmoid(-0.9715693346165644 + -0.9768955137850739*tsigmoid(-0.01594696371419868 + 0.9909497456345888*$(x[1]) + -0.7696063050893072*$(x[2]) + 0.6471253267718304*$(x[3]) + -0.4921765059422283*$(x[4])) + -0.5921792052235291*tsigmoid(0.5214157725539503 + -0.26313320095327697*$(x[1]) + -0.8319081220878042*$(x[2]) + 0.3107042744384958*$(x[3]) + 0.6661282727875562*$(x[4])) + -0.48413001595059546*tsigmoid(0.3231819166326173 + -0.0012088637643752342*$(x[1]) + -0.2674916056933423*$(x[2]) + -0.060297003494380164*$(x[3]) + -0.32592232757567396*$(x[4])) + 0.5958221229003775*tsigmoid(0.5088810466815126 + 0.996358423404454*$(x[1]) + 0.8285410186024262*$(x[2]) + 0.6728266490544299*$(x[3]) + -0.17119992190603783*$(x[4]))) + tsigmoid(0.3174314767171693 + 0.7973516938258234*tsigmoid(-0.01594696371419868 + 0.9909497456345888*$(x[1]) + -0.7696063050893072*$(x[2]) + 0.6471253267718304*$(x[3]) + -0.4921765059422283*$(x[4])) + 0.34613127204807803*tsigmoid(0.5214157725539503 + -0.26313320095327697*$(x[1]) + -0.8319081220878042*$(x[2]) + 0.3107042744384958*$(x[3]) + 0.6661282727875562*$(x[4])) + 0.2873530780915141*tsigmoid(0.3231819166326173 + -0.0012088637643752342*$(x[1]) + -0.2674916056933423*$(x[2]) + -0.060297003494380164*$(x[3]) + -0.32592232757567396*$(x[4])) + -0.022402485514345205*tsigmoid(0.5088810466815126 + 0.996358423404454*$(x[1]) + 0.8285410186024262*$(x[2]) + 0.6728266490544299*$(x[3]) + -0.17119992190603783*$(x[4]))) + tsigmoid(-0.13273008487791982 + -0.6083643579728482*tsigmoid(-0.01594696371419868 + 0.9909497456345888*$(x[1]) + -0.7696063050893072*$(x[2]) + 0.6471253267718304*$(x[3]) + -0.4921765059422283*$(x[4])) + -0.45497938405127814*tsigmoid(0.5214157725539503 + -0.26313320095327697*$(x[1]) + -0.8319081220878042*$(x[2]) + 0.3107042744384958*$(x[3]) + 0.6661282727875562*$(x[4])) + -0.239169006957638*tsigmoid(0.3231819166326173 + -0.0012088637643752342*$(x[1]) + -0.2674916056933423*$(x[2]) + -0.060297003494380164*$(x[3]) + -0.32592232757567396*$(x[4])) + -0.027875380690040252*tsigmoid(0.5088810466815126 + 0.996358423404454*$(x[1]) + 0.8285410186024262*$(x[2]) + 0.6728266490544299*$(x[3]) + -0.17119992190603783*$(x[4]))) + tsigmoid(-0.04682057468269285 + -0.5706267030647423*tsigmoid(-0.01594696371419868 + 0.9909497456345888*$(x[1]) + -0.7696063050893072*$(x[2]) + 0.6471253267718304*$(x[3]) + -0.4921765059422283*$(x[4])) + 0.4232761084933685*tsigmoid(0.5214157725539503 + -0.26313320095327697*$(x[1]) + -0.8319081220878042*$(x[2]) + 0.3107042744384958*$(x[3]) + 0.6661282727875562*$(x[4])) + -0.30221570555967947*tsigmoid(0.3231819166326173 + -0.0012088637643752342*$(x[1]) + -0.2674916056933423*$(x[2]) + -0.060297003494380164*$(x[3]) + -0.32592232757567396*$(x[4])) + -0.688917335446487*tsigmoid(0.5088810466815126 + 0.996358423404454*$(x[1]) + 0.8285410186024262*$(x[2]) + 0.6728266490544299*$(x[3]) + -0.17119992190603783*$(x[4]))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    