using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -3.892086739821988 <= q <= 1.133112254487742)

                     add_NL_constraint(m, :(log(1 + exp(-0.9215225812660202 + -0.8659264200816277*log(1 + exp(0.9754315885766114 + 0.9182647481159649*$(x[1]) + 0.2963572993024828*$(x[2]))) + -0.7655479829179535*log(1 + exp(0.7516798050016731 + -0.4187259277447679*$(x[1]) + -0.991762240929563*$(x[2]))))) + log(1 + exp(0.9152212659843197 + -0.09622429015475253*log(1 + exp(0.9754315885766114 + 0.9182647481159649*$(x[1]) + 0.2963572993024828*$(x[2]))) + 0.18727818139193309*log(1 + exp(0.7516798050016731 + -0.4187259277447679*$(x[1]) + -0.991762240929563*$(x[2]))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    