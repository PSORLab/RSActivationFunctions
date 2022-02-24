using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:4] <= 1)

                     @variable(m, -16.424759279035918 <= q <= 20.23452089871623)

                     add_NL_constraint(m, :(1/(1 + exp(-(0.8420310557117703 + 0.34198141401583015*1/(1 + exp(-(-0.5389217244927917 + 0.5798025520034438*1/(1 + exp(-(0.8931287257144485 + 0.7044247236000336*$(x[1]) + 0.9746565459142618*$(x[2]) + -0.37297510015038515*$(x[3]) + 0.2762326932574686*$(x[4])))) + 0.4428632765655749*1/(1 + exp(-(-0.08155088834170021 + 0.2657824238897022*$(x[1]) + -0.7808635499515542*$(x[2]) + -0.8071446449659851*$(x[3]) + 0.29051627947702796*$(x[4])))) + 0.9033224385695577*1/(1 + exp(-(0.5572058350137663 + -0.028961919981738138*$(x[1]) + -0.957730339736961*$(x[2]) + -0.8718568123342201*$(x[3]) + -0.17639541567112582*$(x[4]))))))) + -0.1887519896617813*1/(1 + exp(-(-0.23921485861490321 + 0.5030319708107833*1/(1 + exp(-(0.8931287257144485 + 0.7044247236000336*$(x[1]) + 0.9746565459142618*$(x[2]) + -0.37297510015038515*$(x[3]) + 0.2762326932574686*$(x[4])))) + -0.2081481501163327*1/(1 + exp(-(-0.08155088834170021 + 0.2657824238897022*$(x[1]) + -0.7808635499515542*$(x[2]) + -0.8071446449659851*$(x[3]) + 0.29051627947702796*$(x[4])))) + -0.2939216839670169*1/(1 + exp(-(0.5572058350137663 + -0.028961919981738138*$(x[1]) + -0.957730339736961*$(x[2]) + -0.8718568123342201*$(x[3]) + -0.17639541567112582*$(x[4]))))))) + -0.2882490310085508*1/(1 + exp(-(-0.6545847926105979 + 0.9774628615021363*1/(1 + exp(-(0.8931287257144485 + 0.7044247236000336*$(x[1]) + 0.9746565459142618*$(x[2]) + -0.37297510015038515*$(x[3]) + 0.2762326932574686*$(x[4])))) + -0.21180887804098658*1/(1 + exp(-(-0.08155088834170021 + 0.2657824238897022*$(x[1]) + -0.7808635499515542*$(x[2]) + -0.8071446449659851*$(x[3]) + 0.29051627947702796*$(x[4])))) + -0.7013494059895882*1/(1 + exp(-(0.5572058350137663 + -0.028961919981738138*$(x[1]) + -0.957730339736961*$(x[2]) + -0.8718568123342201*$(x[3]) + -0.17639541567112582*$(x[4])))))))))) + 1/(1 + exp(-(-0.35188345522271236 + -0.11415509244934263*1/(1 + exp(-(-0.5389217244927917 + 0.5798025520034438*1/(1 + exp(-(0.8931287257144485 + 0.7044247236000336*$(x[1]) + 0.9746565459142618*$(x[2]) + -0.37297510015038515*$(x[3]) + 0.2762326932574686*$(x[4])))) + 0.4428632765655749*1/(1 + exp(-(-0.08155088834170021 + 0.2657824238897022*$(x[1]) + -0.7808635499515542*$(x[2]) + -0.8071446449659851*$(x[3]) + 0.29051627947702796*$(x[4])))) + 0.9033224385695577*1/(1 + exp(-(0.5572058350137663 + -0.028961919981738138*$(x[1]) + -0.957730339736961*$(x[2]) + -0.8718568123342201*$(x[3]) + -0.17639541567112582*$(x[4]))))))) + -0.8088728193024375*1/(1 + exp(-(-0.23921485861490321 + 0.5030319708107833*1/(1 + exp(-(0.8931287257144485 + 0.7044247236000336*$(x[1]) + 0.9746565459142618*$(x[2]) + -0.37297510015038515*$(x[3]) + 0.2762326932574686*$(x[4])))) + -0.2081481501163327*1/(1 + exp(-(-0.08155088834170021 + 0.2657824238897022*$(x[1]) + -0.7808635499515542*$(x[2]) + -0.8071446449659851*$(x[3]) + 0.29051627947702796*$(x[4])))) + -0.2939216839670169*1/(1 + exp(-(0.5572058350137663 + -0.028961919981738138*$(x[1]) + -0.957730339736961*$(x[2]) + -0.8718568123342201*$(x[3]) + -0.17639541567112582*$(x[4]))))))) + 0.8271426964720727*1/(1 + exp(-(-0.6545847926105979 + 0.9774628615021363*1/(1 + exp(-(0.8931287257144485 + 0.7044247236000336*$(x[1]) + 0.9746565459142618*$(x[2]) + -0.37297510015038515*$(x[3]) + 0.2762326932574686*$(x[4])))) + -0.21180887804098658*1/(1 + exp(-(-0.08155088834170021 + 0.2657824238897022*$(x[1]) + -0.7808635499515542*$(x[2]) + -0.8071446449659851*$(x[3]) + 0.29051627947702796*$(x[4])))) + -0.7013494059895882*1/(1 + exp(-(0.5572058350137663 + -0.028961919981738138*$(x[1]) + -0.957730339736961*$(x[2]) + -0.8718568123342201*$(x[3]) + -0.17639541567112582*$(x[4])))))))))) + 1/(1 + exp(-(0.4798469380896484 + 0.8134725364327324*1/(1 + exp(-(-0.5389217244927917 + 0.5798025520034438*1/(1 + exp(-(0.8931287257144485 + 0.7044247236000336*$(x[1]) + 0.9746565459142618*$(x[2]) + -0.37297510015038515*$(x[3]) + 0.2762326932574686*$(x[4])))) + 0.4428632765655749*1/(1 + exp(-(-0.08155088834170021 + 0.2657824238897022*$(x[1]) + -0.7808635499515542*$(x[2]) + -0.8071446449659851*$(x[3]) + 0.29051627947702796*$(x[4])))) + 0.9033224385695577*1/(1 + exp(-(0.5572058350137663 + -0.028961919981738138*$(x[1]) + -0.957730339736961*$(x[2]) + -0.8718568123342201*$(x[3]) + -0.17639541567112582*$(x[4]))))))) + -0.12585836968848696*1/(1 + exp(-(-0.23921485861490321 + 0.5030319708107833*1/(1 + exp(-(0.8931287257144485 + 0.7044247236000336*$(x[1]) + 0.9746565459142618*$(x[2]) + -0.37297510015038515*$(x[3]) + 0.2762326932574686*$(x[4])))) + -0.2081481501163327*1/(1 + exp(-(-0.08155088834170021 + 0.2657824238897022*$(x[1]) + -0.7808635499515542*$(x[2]) + -0.8071446449659851*$(x[3]) + 0.29051627947702796*$(x[4])))) + -0.2939216839670169*1/(1 + exp(-(0.5572058350137663 + -0.028961919981738138*$(x[1]) + -0.957730339736961*$(x[2]) + -0.8718568123342201*$(x[3]) + -0.17639541567112582*$(x[4]))))))) + 0.613011911779838*1/(1 + exp(-(-0.6545847926105979 + 0.9774628615021363*1/(1 + exp(-(0.8931287257144485 + 0.7044247236000336*$(x[1]) + 0.9746565459142618*$(x[2]) + -0.37297510015038515*$(x[3]) + 0.2762326932574686*$(x[4])))) + -0.21180887804098658*1/(1 + exp(-(-0.08155088834170021 + 0.2657824238897022*$(x[1]) + -0.7808635499515542*$(x[2]) + -0.8071446449659851*$(x[3]) + 0.29051627947702796*$(x[4])))) + -0.7013494059895882*1/(1 + exp(-(0.5572058350137663 + -0.028961919981738138*$(x[1]) + -0.957730339736961*$(x[2]) + -0.8718568123342201*$(x[3]) + -0.17639541567112582*$(x[4])))))))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    