using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -7.107954588851326 <= q <= 8.439913276188518)

                     add_NL_constraint(m, :(softplus(-0.4953615570310643 + -0.5197659840811282*softplus(0.7941458217392698 + 0.49653685598476827*softplus(-0.2081348932235647 + -0.9819593709419507*$(x[1]) + 0.4754911924134797*$(x[2]) + -0.6006888917392823*$(x[3])) + 0.8533482485058403*softplus(0.14812040410194793 + -0.8433474401842953*$(x[1]) + -0.1761112323855083*$(x[2]) + 0.6144836466445156*$(x[3])) + -0.13525586788633204*softplus(-0.3937152244783322 + 0.36708048782945024*$(x[1]) + 0.138776295405171*$(x[2]) + -0.8492811105063662*$(x[3]))) + -0.07122185861106178*softplus(-0.06420985369672083 + 0.9131512287471262*softplus(-0.2081348932235647 + -0.9819593709419507*$(x[1]) + 0.4754911924134797*$(x[2]) + -0.6006888917392823*$(x[3])) + 0.03263883937508805*softplus(0.14812040410194793 + -0.8433474401842953*$(x[1]) + -0.1761112323855083*$(x[2]) + 0.6144836466445156*$(x[3])) + 0.0015134736925910275*softplus(-0.3937152244783322 + 0.36708048782945024*$(x[1]) + 0.138776295405171*$(x[2]) + -0.8492811105063662*$(x[3]))) + 0.004105284291951339*softplus(-0.8124442669202567 + 0.4437121866105662*softplus(-0.2081348932235647 + -0.9819593709419507*$(x[1]) + 0.4754911924134797*$(x[2]) + -0.6006888917392823*$(x[3])) + 0.26832550078458484*softplus(0.14812040410194793 + -0.8433474401842953*$(x[1]) + -0.1761112323855083*$(x[2]) + 0.6144836466445156*$(x[3])) + 0.2804882536336004*softplus(-0.3937152244783322 + 0.36708048782945024*$(x[1]) + 0.138776295405171*$(x[2]) + -0.8492811105063662*$(x[3])))) + softplus(0.7344840166166842 + -0.5219551557665656*softplus(0.7941458217392698 + 0.49653685598476827*softplus(-0.2081348932235647 + -0.9819593709419507*$(x[1]) + 0.4754911924134797*$(x[2]) + -0.6006888917392823*$(x[3])) + 0.8533482485058403*softplus(0.14812040410194793 + -0.8433474401842953*$(x[1]) + -0.1761112323855083*$(x[2]) + 0.6144836466445156*$(x[3])) + -0.13525586788633204*softplus(-0.3937152244783322 + 0.36708048782945024*$(x[1]) + 0.138776295405171*$(x[2]) + -0.8492811105063662*$(x[3]))) + -0.12237894168399821*softplus(-0.06420985369672083 + 0.9131512287471262*softplus(-0.2081348932235647 + -0.9819593709419507*$(x[1]) + 0.4754911924134797*$(x[2]) + -0.6006888917392823*$(x[3])) + 0.03263883937508805*softplus(0.14812040410194793 + -0.8433474401842953*$(x[1]) + -0.1761112323855083*$(x[2]) + 0.6144836466445156*$(x[3])) + 0.0015134736925910275*softplus(-0.3937152244783322 + 0.36708048782945024*$(x[1]) + 0.138776295405171*$(x[2]) + -0.8492811105063662*$(x[3]))) + 0.5945370485807278*softplus(-0.8124442669202567 + 0.4437121866105662*softplus(-0.2081348932235647 + -0.9819593709419507*$(x[1]) + 0.4754911924134797*$(x[2]) + -0.6006888917392823*$(x[3])) + 0.26832550078458484*softplus(0.14812040410194793 + -0.8433474401842953*$(x[1]) + -0.1761112323855083*$(x[2]) + 0.6144836466445156*$(x[3])) + 0.2804882536336004*softplus(-0.3937152244783322 + 0.36708048782945024*$(x[1]) + 0.138776295405171*$(x[2]) + -0.8492811105063662*$(x[3])))) + softplus(-0.7546740103129919 + -0.5743271753296177*softplus(0.7941458217392698 + 0.49653685598476827*softplus(-0.2081348932235647 + -0.9819593709419507*$(x[1]) + 0.4754911924134797*$(x[2]) + -0.6006888917392823*$(x[3])) + 0.8533482485058403*softplus(0.14812040410194793 + -0.8433474401842953*$(x[1]) + -0.1761112323855083*$(x[2]) + 0.6144836466445156*$(x[3])) + -0.13525586788633204*softplus(-0.3937152244783322 + 0.36708048782945024*$(x[1]) + 0.138776295405171*$(x[2]) + -0.8492811105063662*$(x[3]))) + 0.959961363428667*softplus(-0.06420985369672083 + 0.9131512287471262*softplus(-0.2081348932235647 + -0.9819593709419507*$(x[1]) + 0.4754911924134797*$(x[2]) + -0.6006888917392823*$(x[3])) + 0.03263883937508805*softplus(0.14812040410194793 + -0.8433474401842953*$(x[1]) + -0.1761112323855083*$(x[2]) + 0.6144836466445156*$(x[3])) + 0.0015134736925910275*softplus(-0.3937152244783322 + 0.36708048782945024*$(x[1]) + 0.138776295405171*$(x[2]) + -0.8492811105063662*$(x[3]))) + -0.0685664132974031*softplus(-0.8124442669202567 + 0.4437121866105662*softplus(-0.2081348932235647 + -0.9819593709419507*$(x[1]) + 0.4754911924134797*$(x[2]) + -0.6006888917392823*$(x[3])) + 0.26832550078458484*softplus(0.14812040410194793 + -0.8433474401842953*$(x[1]) + -0.1761112323855083*$(x[2]) + 0.6144836466445156*$(x[3])) + 0.2804882536336004*softplus(-0.3937152244783322 + 0.36708048782945024*$(x[1]) + 0.138776295405171*$(x[2]) + -0.8492811105063662*$(x[3])))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    