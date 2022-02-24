using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -7.9284408379108005 <= q <= 14.377266175441108)

                     add_NL_constraint(m, :(swish(0.4343485145630286 + -0.5735104656090533*swish(-0.47832195115882303 + 0.6358617391786257*$(x[1]) + -0.4037373919488174*$(x[2])) + -0.40833038450807946*swish(0.4113265195541662 + -0.7994001067331897*$(x[1]) + -0.3401124060181533*$(x[2])) + 0.09714914815406805*swish(-0.3249145933093094 + 0.9313960721440679*$(x[1]) + 0.2643917347767659*$(x[2])) + -0.19623354791767778*swish(0.34085895078435247 + -0.33650259644885194*$(x[1]) + -0.9436140496760688*$(x[2])) + -0.3237276355524821*swish(0.8342164127508305 + -0.6721074785776127*$(x[1]) + 0.009028597341863787*$(x[2]))) + swish(-0.11905941659917563 + 0.17224590981590104*swish(-0.47832195115882303 + 0.6358617391786257*$(x[1]) + -0.4037373919488174*$(x[2])) + 0.8406566605626429*swish(0.4113265195541662 + -0.7994001067331897*$(x[1]) + -0.3401124060181533*$(x[2])) + 0.24719777641241691*swish(-0.3249145933093094 + 0.9313960721440679*$(x[1]) + 0.2643917347767659*$(x[2])) + -0.24308200319021767*swish(0.34085895078435247 + -0.33650259644885194*$(x[1]) + -0.9436140496760688*$(x[2])) + -0.39557038452022475*swish(0.8342164127508305 + -0.6721074785776127*$(x[1]) + 0.009028597341863787*$(x[2]))) + swish(0.967203829156619 + 0.3785230261408685*swish(-0.47832195115882303 + 0.6358617391786257*$(x[1]) + -0.4037373919488174*$(x[2])) + -0.23118999327456224*swish(0.4113265195541662 + -0.7994001067331897*$(x[1]) + -0.3401124060181533*$(x[2])) + -0.5197073892915105*swish(-0.3249145933093094 + 0.9313960721440679*$(x[1]) + 0.2643917347767659*$(x[2])) + 0.6005003791404913*swish(0.34085895078435247 + -0.33650259644885194*$(x[1]) + -0.9436140496760688*$(x[2])) + 0.15463985115916623*swish(0.8342164127508305 + -0.6721074785776127*$(x[1]) + 0.009028597341863787*$(x[2]))) + swish(-0.18340968019959902 + 0.01980109053282808*swish(-0.47832195115882303 + 0.6358617391786257*$(x[1]) + -0.4037373919488174*$(x[2])) + 0.8767310690037822*swish(0.4113265195541662 + -0.7994001067331897*$(x[1]) + -0.3401124060181533*$(x[2])) + 0.42243485881005904*swish(-0.3249145933093094 + 0.9313960721440679*$(x[1]) + 0.2643917347767659*$(x[2])) + -0.1694649191201152*swish(0.34085895078435247 + -0.33650259644885194*$(x[1]) + -0.9436140496760688*$(x[2])) + 0.5278959282338329*swish(0.8342164127508305 + -0.6721074785776127*$(x[1]) + 0.009028597341863787*$(x[2]))) + swish(0.7043455967116397 + -0.48930630508951234*swish(-0.47832195115882303 + 0.6358617391786257*$(x[1]) + -0.4037373919488174*$(x[2])) + 0.7375930151357304*swish(0.4113265195541662 + -0.7994001067331897*$(x[1]) + -0.3401124060181533*$(x[2])) + -0.9188673755139103*swish(-0.3249145933093094 + 0.9313960721440679*$(x[1]) + 0.2643917347767659*$(x[2])) + 0.5801272291202126*swish(0.34085895078435247 + -0.33650259644885194*$(x[1]) + -0.9436140496760688*$(x[2])) + 0.06743755305487698*swish(0.8342164127508305 + -0.6721074785776127*$(x[1]) + 0.009028597341863787*$(x[2]))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    