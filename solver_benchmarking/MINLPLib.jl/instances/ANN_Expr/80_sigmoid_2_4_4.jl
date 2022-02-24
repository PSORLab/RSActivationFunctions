using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -47.221875796417265 <= q <= 49.11122836315772)

                     add_NL_constraint(m, :(1/(1 + exp(-(0.42790162605340454 + 0.9948146339173967*1/(1 + exp(-(0.30429674417798 + 0.01326158862793747*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.8506753528903843*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.7699605184902198*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.9481609482474158*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2])))))))))) + 0.5690258167368487*1/(1 + exp(-(0.4003333530904172 + 0.4565361301167439*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + -0.41218949132242644*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.6719961553691296*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.9454422589116072*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2])))))))))) + -0.632632285898699*1/(1 + exp(-(-0.4740061987673023 + -0.21503151130953757*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.7641417558802628*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.8796988576372375*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.6466469730144713*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2])))))))))) + -0.0799939753544896*1/(1 + exp(-(-0.9533597279323933 + 0.019939841305541606*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + -0.8134733780415981*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + -0.10348468441851777*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.008542712959329624*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))))))))) + 1/(1 + exp(-(-0.31282642357163937 + 0.400776902433146*1/(1 + exp(-(0.30429674417798 + 0.01326158862793747*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.8506753528903843*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.7699605184902198*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.9481609482474158*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2])))))))))) + -0.8700113611274283*1/(1 + exp(-(0.4003333530904172 + 0.4565361301167439*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + -0.41218949132242644*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.6719961553691296*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.9454422589116072*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2])))))))))) + -0.8173535730312884*1/(1 + exp(-(-0.4740061987673023 + -0.21503151130953757*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.7641417558802628*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.8796988576372375*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.6466469730144713*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2])))))))))) + -0.10383649475667989*1/(1 + exp(-(-0.9533597279323933 + 0.019939841305541606*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + -0.8134733780415981*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + -0.10348468441851777*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.008542712959329624*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))))))))) + 1/(1 + exp(-(0.07748802701261592 + -0.48287575350847955*1/(1 + exp(-(0.30429674417798 + 0.01326158862793747*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.8506753528903843*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.7699605184902198*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.9481609482474158*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2])))))))))) + 0.9032398979921337*1/(1 + exp(-(0.4003333530904172 + 0.4565361301167439*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + -0.41218949132242644*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.6719961553691296*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.9454422589116072*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2])))))))))) + 0.8154202752057489*1/(1 + exp(-(-0.4740061987673023 + -0.21503151130953757*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.7641417558802628*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.8796988576372375*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.6466469730144713*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2])))))))))) + 0.8885160528665201*1/(1 + exp(-(-0.9533597279323933 + 0.019939841305541606*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + -0.8134733780415981*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + -0.10348468441851777*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.008542712959329624*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))))))))) + 1/(1 + exp(-(0.6389491685818425 + -0.9024035564379238*1/(1 + exp(-(0.30429674417798 + 0.01326158862793747*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.8506753528903843*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.7699605184902198*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.9481609482474158*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2])))))))))) + -0.8688708754259227*1/(1 + exp(-(0.4003333530904172 + 0.4565361301167439*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + -0.41218949132242644*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.6719961553691296*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.9454422589116072*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2])))))))))) + -0.9399903753655909*1/(1 + exp(-(-0.4740061987673023 + -0.21503151130953757*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.7641417558802628*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.8796988576372375*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.6466469730144713*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2])))))))))) + 0.5656846187923494*1/(1 + exp(-(-0.9533597279323933 + 0.019939841305541606*1/(1 + exp(-(0.7020479917236249 + 0.5157446936496526*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.8162922098274223*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6507571114046486*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + 0.3903970834590602*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + -0.8134733780415981*1/(1 + exp(-(0.7594455380381384 + 0.9088497207479063*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + 0.34867162336420376*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + -0.586942758624089*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.05285512797069236*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + -0.10348468441851777*1/(1 + exp(-(0.5333438081093944 + -0.9884706408297048*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.9213430875393112*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.6584439947423988*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.39397409418232554*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))) + 0.008542712959329624*1/(1 + exp(-(-0.8688273038537231 + -0.5762764679031651*1/(1 + exp(-(-0.990780872931377 + 0.7098282678057655*$(x[1]) + -0.9123645360112271*$(x[2])))) + -0.4115583086286181*1/(1 + exp(-(-0.26001558800187974 + -0.2578347550825866*$(x[1]) + -0.8440952761970419*$(x[2])))) + 0.07598378122288318*1/(1 + exp(-(0.6359235954642828 + 0.0835591752979501*$(x[1]) + -0.4064923799931912*$(x[2])))) + -0.5005567670062852*1/(1 + exp(-(0.919513762198326 + -0.48206842436164976*$(x[1]) + 0.2512150299638187*$(x[2]))))))))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    