using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -16.533362936959833 <= q <= 10.809894720194647)

                     add_NL_constraint(m, :(swish(-0.37520834136035575 + 0.9494325570989135*swish(-0.28211052244792345 + -0.99806857870874*swish(-0.8742373404120025 + -0.426265240232806*$(x[1]) + -0.3837227572332127*$(x[2]) + -0.8111724363793971*$(x[3])) + -0.5868052369549814*swish(0.1820521845200731 + -0.21711855296330995*$(x[1]) + 0.06386620198486614*$(x[2]) + -0.46598761690463464*$(x[3])) + -0.7519368767825174*swish(0.131258377923118 + -0.25079688309125725*$(x[1]) + -0.2597606256154239*$(x[2]) + -0.38875983122465607*$(x[3]))) + -0.48600023069247467*swish(-0.36512748866340017 + -0.9861061514138418*swish(-0.8742373404120025 + -0.426265240232806*$(x[1]) + -0.3837227572332127*$(x[2]) + -0.8111724363793971*$(x[3])) + 0.24795063763871195*swish(0.1820521845200731 + -0.21711855296330995*$(x[1]) + 0.06386620198486614*$(x[2]) + -0.46598761690463464*$(x[3])) + 0.26423028298383944*swish(0.131258377923118 + -0.25079688309125725*$(x[1]) + -0.2597606256154239*$(x[2]) + -0.38875983122465607*$(x[3]))) + 0.2786638895614333*swish(-0.06661892073119935 + -0.8857405042008271*swish(-0.8742373404120025 + -0.426265240232806*$(x[1]) + -0.3837227572332127*$(x[2]) + -0.8111724363793971*$(x[3])) + -0.7387399407812776*swish(0.1820521845200731 + -0.21711855296330995*$(x[1]) + 0.06386620198486614*$(x[2]) + -0.46598761690463464*$(x[3])) + 0.20379172616564079*swish(0.131258377923118 + -0.25079688309125725*$(x[1]) + -0.2597606256154239*$(x[2]) + -0.38875983122465607*$(x[3])))) + swish(-0.7730743089363572 + 0.3681741825242679*swish(-0.28211052244792345 + -0.99806857870874*swish(-0.8742373404120025 + -0.426265240232806*$(x[1]) + -0.3837227572332127*$(x[2]) + -0.8111724363793971*$(x[3])) + -0.5868052369549814*swish(0.1820521845200731 + -0.21711855296330995*$(x[1]) + 0.06386620198486614*$(x[2]) + -0.46598761690463464*$(x[3])) + -0.7519368767825174*swish(0.131258377923118 + -0.25079688309125725*$(x[1]) + -0.2597606256154239*$(x[2]) + -0.38875983122465607*$(x[3]))) + -0.6127984670266708*swish(-0.36512748866340017 + -0.9861061514138418*swish(-0.8742373404120025 + -0.426265240232806*$(x[1]) + -0.3837227572332127*$(x[2]) + -0.8111724363793971*$(x[3])) + 0.24795063763871195*swish(0.1820521845200731 + -0.21711855296330995*$(x[1]) + 0.06386620198486614*$(x[2]) + -0.46598761690463464*$(x[3])) + 0.26423028298383944*swish(0.131258377923118 + -0.25079688309125725*$(x[1]) + -0.2597606256154239*$(x[2]) + -0.38875983122465607*$(x[3]))) + -0.09266796191651139*swish(-0.06661892073119935 + -0.8857405042008271*swish(-0.8742373404120025 + -0.426265240232806*$(x[1]) + -0.3837227572332127*$(x[2]) + -0.8111724363793971*$(x[3])) + -0.7387399407812776*swish(0.1820521845200731 + -0.21711855296330995*$(x[1]) + 0.06386620198486614*$(x[2]) + -0.46598761690463464*$(x[3])) + 0.20379172616564079*swish(0.131258377923118 + -0.25079688309125725*$(x[1]) + -0.2597606256154239*$(x[2]) + -0.38875983122465607*$(x[3])))) + swish(0.18309763888185682 + -0.5322392313183273*swish(-0.28211052244792345 + -0.99806857870874*swish(-0.8742373404120025 + -0.426265240232806*$(x[1]) + -0.3837227572332127*$(x[2]) + -0.8111724363793971*$(x[3])) + -0.5868052369549814*swish(0.1820521845200731 + -0.21711855296330995*$(x[1]) + 0.06386620198486614*$(x[2]) + -0.46598761690463464*$(x[3])) + -0.7519368767825174*swish(0.131258377923118 + -0.25079688309125725*$(x[1]) + -0.2597606256154239*$(x[2]) + -0.38875983122465607*$(x[3]))) + 0.5415916403662484*swish(-0.36512748866340017 + -0.9861061514138418*swish(-0.8742373404120025 + -0.426265240232806*$(x[1]) + -0.3837227572332127*$(x[2]) + -0.8111724363793971*$(x[3])) + 0.24795063763871195*swish(0.1820521845200731 + -0.21711855296330995*$(x[1]) + 0.06386620198486614*$(x[2]) + -0.46598761690463464*$(x[3])) + 0.26423028298383944*swish(0.131258377923118 + -0.25079688309125725*$(x[1]) + -0.2597606256154239*$(x[2]) + -0.38875983122465607*$(x[3]))) + -0.5996499854211454*swish(-0.06661892073119935 + -0.8857405042008271*swish(-0.8742373404120025 + -0.426265240232806*$(x[1]) + -0.3837227572332127*$(x[2]) + -0.8111724363793971*$(x[3])) + -0.7387399407812776*swish(0.1820521845200731 + -0.21711855296330995*$(x[1]) + 0.06386620198486614*$(x[2]) + -0.46598761690463464*$(x[3])) + 0.20379172616564079*swish(0.131258377923118 + -0.25079688309125725*$(x[1]) + -0.2597606256154239*$(x[2]) + -0.38875983122465607*$(x[3])))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    