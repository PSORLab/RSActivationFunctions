using JuMP, EAGO

                     m = Model()

                     register(m, :tswish, 1, tswish, autodiff = true)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -6.394911425236015 <= q <= 5.924457438474725)

                     add_NL_constraint(m, :(tswish(0.7412515189461533 + -0.259123025038706*$(x[1]) + 0.7519639806952378*$(x[2]) + 0.26999869846688185*$(x[3]) + -0.8297445972809041*$(x[4])) + tswish(-0.20402168514124064 + 0.4681214549834145*$(x[1]) + -0.37495490398828313*$(x[2]) + -0.48838629994892946*$(x[3]) + 0.32029417302406404*$(x[4])) + tswish(-0.7724568271855574 + -0.37109258631268327*$(x[1]) + 0.8050444913099231*$(x[2]) + -0.4992168112771722*$(x[3]) + 0.7217434095291702*$(x[4])) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    