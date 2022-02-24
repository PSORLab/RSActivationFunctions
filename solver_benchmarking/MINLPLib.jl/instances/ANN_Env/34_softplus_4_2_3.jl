using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -6.019960436071491 <= q <= 11.696817216382863)

                     add_NL_constraint(m, :(softplus(0.9042935565441375 + 0.31236426273908524*softplus(0.5868252904954097 + 0.04712499510197299*$(x[1]) + 0.4005456947892885*$(x[2]) + 0.34544434364772814*$(x[3]) + 0.48957180414623247*$(x[4])) + -0.4128166127064836*softplus(0.5797949667252156 + -0.7245574143823843*$(x[1]) + -0.08207866517990237*$(x[2]) + 0.4631961331947827*$(x[3]) + 0.9179781125579716*$(x[4])) + -0.2476169775927386*softplus(0.8773151656273224 + 0.003673243009740812*$(x[1]) + 0.698013040735662*$(x[2]) + 0.6787695040522581*$(x[3]) + -0.9128403322093082*$(x[4]))) + softplus(0.8600467319105785 + -0.4006275013725942*softplus(0.5868252904954097 + 0.04712499510197299*$(x[1]) + 0.4005456947892885*$(x[2]) + 0.34544434364772814*$(x[3]) + 0.48957180414623247*$(x[4])) + 0.22106887260345687*softplus(0.5797949667252156 + -0.7245574143823843*$(x[1]) + -0.08207866517990237*$(x[2]) + 0.4631961331947827*$(x[3]) + 0.9179781125579716*$(x[4])) + 0.7210213948433983*softplus(0.8773151656273224 + 0.003673243009740812*$(x[1]) + 0.698013040735662*$(x[2]) + 0.6787695040522581*$(x[3]) + -0.9128403322093082*$(x[4]))) + softplus(0.982756228863964 + 0.34792310275142935*softplus(0.5868252904954097 + 0.04712499510197299*$(x[1]) + 0.4005456947892885*$(x[2]) + 0.34544434364772814*$(x[3]) + 0.48957180414623247*$(x[4])) + 0.7924515917531365*softplus(0.5797949667252156 + -0.7245574143823843*$(x[1]) + -0.08207866517990237*$(x[2]) + 0.4631961331947827*$(x[3]) + 0.9179781125579716*$(x[4])) + -0.9399735455074207*softplus(0.8773151656273224 + 0.003673243009740812*$(x[1]) + 0.698013040735662*$(x[2]) + 0.6787695040522581*$(x[3]) + -0.9128403322093082*$(x[4]))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    