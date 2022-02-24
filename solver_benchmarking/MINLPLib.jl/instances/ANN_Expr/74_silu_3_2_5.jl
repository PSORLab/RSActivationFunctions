using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -19.736879506320445 <= q <= 22.217442927624916)

                     add_NL_constraint(m, :((-0.3136061325511519 + 0.2821341474705279*(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3]))/(1 + exp(-(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3])))) + 0.4396178028948987*(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3]))/(1 + exp(-(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3])))) + -0.24849525966922226*(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3]))/(1 + exp(-(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3])))) + 0.5972646324288218*(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3]))/(1 + exp(-(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3])))) + -0.8620507270647968*(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))/(1 + exp(-(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3])))))/(1 + exp(-(-0.3136061325511519 + 0.2821341474705279*(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3]))/(1 + exp(-(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3])))) + 0.4396178028948987*(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3]))/(1 + exp(-(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3])))) + -0.24849525966922226*(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3]))/(1 + exp(-(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3])))) + 0.5972646324288218*(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3]))/(1 + exp(-(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3])))) + -0.8620507270647968*(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))/(1 + exp(-(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))))))) + (-0.9705953045797062 + -0.6259685253929317*(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3]))/(1 + exp(-(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3])))) + 0.41918630672941015*(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3]))/(1 + exp(-(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3])))) + -0.9388772793667424*(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3]))/(1 + exp(-(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3])))) + -0.18849282727638839*(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3]))/(1 + exp(-(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3])))) + -0.7819718693068158*(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))/(1 + exp(-(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3])))))/(1 + exp(-(-0.9705953045797062 + -0.6259685253929317*(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3]))/(1 + exp(-(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3])))) + 0.41918630672941015*(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3]))/(1 + exp(-(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3])))) + -0.9388772793667424*(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3]))/(1 + exp(-(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3])))) + -0.18849282727638839*(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3]))/(1 + exp(-(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3])))) + -0.7819718693068158*(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))/(1 + exp(-(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))))))) + (0.8285261526161638 + 0.6863822914423507*(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3]))/(1 + exp(-(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3])))) + -0.7060930690780838*(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3]))/(1 + exp(-(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3])))) + -0.2956343589188948*(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3]))/(1 + exp(-(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3])))) + 0.32353098534638214*(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3]))/(1 + exp(-(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3])))) + 0.6963022628505415*(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))/(1 + exp(-(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3])))))/(1 + exp(-(0.8285261526161638 + 0.6863822914423507*(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3]))/(1 + exp(-(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3])))) + -0.7060930690780838*(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3]))/(1 + exp(-(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3])))) + -0.2956343589188948*(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3]))/(1 + exp(-(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3])))) + 0.32353098534638214*(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3]))/(1 + exp(-(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3])))) + 0.6963022628505415*(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))/(1 + exp(-(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))))))) + (-0.2246585184074279 + 0.8591320330102441*(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3]))/(1 + exp(-(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3])))) + -0.5037522661683527*(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3]))/(1 + exp(-(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3])))) + 0.9621050832972355*(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3]))/(1 + exp(-(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3])))) + -0.1995700954953512*(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3]))/(1 + exp(-(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3])))) + 0.442251462608958*(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))/(1 + exp(-(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3])))))/(1 + exp(-(-0.2246585184074279 + 0.8591320330102441*(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3]))/(1 + exp(-(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3])))) + -0.5037522661683527*(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3]))/(1 + exp(-(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3])))) + 0.9621050832972355*(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3]))/(1 + exp(-(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3])))) + -0.1995700954953512*(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3]))/(1 + exp(-(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3])))) + 0.442251462608958*(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))/(1 + exp(-(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))))))) + (0.5761249271362421 + -0.042724357144423575*(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3]))/(1 + exp(-(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3])))) + -0.4537006876708345*(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3]))/(1 + exp(-(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3])))) + -0.12519521024691826*(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3]))/(1 + exp(-(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3])))) + -0.7426009177864223*(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3]))/(1 + exp(-(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3])))) + -0.8758186284143128*(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))/(1 + exp(-(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3])))))/(1 + exp(-(0.5761249271362421 + -0.042724357144423575*(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3]))/(1 + exp(-(0.7833824073251248 + 0.1390648465658577*$(x[1]) + -0.6617731040629309*$(x[2]) + -0.5974574624055107*$(x[3])))) + -0.4537006876708345*(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3]))/(1 + exp(-(-0.7125949557702831 + 0.9109805514170546*$(x[1]) + 0.42115237771853087*$(x[2]) + 0.7134573642319126*$(x[3])))) + -0.12519521024691826*(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3]))/(1 + exp(-(-0.36270963442709325 + 0.07407412170431016*$(x[1]) + -0.6239570881148437*$(x[2]) + -0.6542866380333012*$(x[3])))) + -0.7426009177864223*(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3]))/(1 + exp(-(-0.5451890615285553 + -0.19626106399785304*$(x[1]) + 0.5284606976702149*$(x[2]) + 0.2694706847237054*$(x[3])))) + -0.8758186284143128*(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))/(1 + exp(-(0.35158020081654806 + 0.8300503955389886*$(x[1]) + -0.8477304064925764*$(x[2]) + 0.1840810006485709*$(x[3]))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    