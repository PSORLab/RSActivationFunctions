using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -29.23432407510253 <= q <= 25.005662503891426)

                     add_NL_constraint(m, :(log(1 + exp(-0.9584800931585375 + 0.8291526833328144*log(1 + exp(0.45833851557375827 + -0.07893784992449904*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.05269389376863742*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.9157122134504139*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.14234137821989812*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))) + -0.47308336785041494*log(1 + exp(0.8182392255084516 + 0.5838043616725095*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.4648893036822681*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.819336300807536*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.1444146117738878*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))) + 0.6108443940953072*log(1 + exp(-0.7015763069538381 + -0.6574215519391942*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.13767395548269734*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.21605537563661414*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.3991258949611094*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))) + -0.4993930565410927*log(1 + exp(-0.21931284624450997 + 0.9567749529797225*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.8630566689068395*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.8170122079846647*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.11471605851894884*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))))) + log(1 + exp(0.475745519197857 + -0.20728788527253217*log(1 + exp(0.45833851557375827 + -0.07893784992449904*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.05269389376863742*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.9157122134504139*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.14234137821989812*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))) + 0.5536591664216424*log(1 + exp(0.8182392255084516 + 0.5838043616725095*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.4648893036822681*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.819336300807536*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.1444146117738878*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))) + 0.5772469178059145*log(1 + exp(-0.7015763069538381 + -0.6574215519391942*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.13767395548269734*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.21605537563661414*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.3991258949611094*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))) + 0.14258534816037427*log(1 + exp(-0.21931284624450997 + 0.9567749529797225*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.8630566689068395*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.8170122079846647*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.11471605851894884*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))))) + log(1 + exp(-0.19885854025662475 + 0.9363781836864602*log(1 + exp(0.45833851557375827 + -0.07893784992449904*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.05269389376863742*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.9157122134504139*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.14234137821989812*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))) + -0.2868856241387201*log(1 + exp(0.8182392255084516 + 0.5838043616725095*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.4648893036822681*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.819336300807536*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.1444146117738878*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))) + 0.6989952318088708*log(1 + exp(-0.7015763069538381 + -0.6574215519391942*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.13767395548269734*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.21605537563661414*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.3991258949611094*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))) + 0.2430276118759056*log(1 + exp(-0.21931284624450997 + 0.9567749529797225*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.8630566689068395*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.8170122079846647*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.11471605851894884*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))))) + log(1 + exp(-0.4748268724301301 + -0.943475605090133*log(1 + exp(0.45833851557375827 + -0.07893784992449904*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.05269389376863742*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.9157122134504139*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.14234137821989812*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))) + -0.504486324487766*log(1 + exp(0.8182392255084516 + 0.5838043616725095*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.4648893036822681*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.819336300807536*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.1444146117738878*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))) + -0.8621266685661562*log(1 + exp(-0.7015763069538381 + -0.6574215519391942*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.13767395548269734*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.21605537563661414*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.3991258949611094*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))) + -0.8772206856506037*log(1 + exp(-0.21931284624450997 + 0.9567749529797225*log(1 + exp(-0.22994750449881307 + -0.3590609593879259*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.10337862983418544*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.321850031280575*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.612120130564894*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.8630566689068395*log(1 + exp(0.19384674006188307 + -0.9239566552941776*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + -0.4571782009479768*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + -0.7401514915717242*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.3444562251116623*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + 0.8170122079846647*log(1 + exp(0.4065836733363821 + 0.7200826994240335*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.11568748742914314*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.5958961905666036*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + -0.0402986153245215*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))) + -0.11471605851894884*log(1 + exp(-0.18406387526761447 + 0.28290146144347617*log(1 + exp(-0.9795230057859095 + -0.9823842606294351*$(x[1]) + 0.3621989809961321*$(x[2]))) + 0.5991916639069861*log(1 + exp(0.2857383741568795 + 0.17154081707188507*$(x[1]) + -0.2093148258397095*$(x[2]))) + 0.7799932719867426*log(1 + exp(-0.5451083144913764 + -0.518481685587207*$(x[1]) + 0.024249687825614874*$(x[2]))) + 0.858692422726886*log(1 + exp(-0.02703791931361943 + -0.7859299073841055*$(x[1]) + 0.7832525238663837*$(x[2]))))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    