using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -0.9705716506753843 <= q <= 1.6199082189427347)

                     add_NL_constraint(m, :((0.27812717705086953 + -0.4967460082986306*(-0.6740257057257448 + -0.0007917455797805673*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.5330540695702357*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((-0.6740257057257448 + -0.0007917455797805673*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.5330540695702357*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2 + -0.883304587160544*(0.3709718635886192 + -0.47514710764751555*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.6627028336583534*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((0.3709718635886192 + -0.47514710764751555*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.6627028336583534*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((0.27812717705086953 + -0.4967460082986306*(-0.6740257057257448 + -0.0007917455797805673*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.5330540695702357*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((-0.6740257057257448 + -0.0007917455797805673*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.5330540695702357*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2 + -0.883304587160544*(0.3709718635886192 + -0.47514710764751555*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.6627028336583534*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((0.3709718635886192 + -0.47514710764751555*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.6627028336583534*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2 + (0.48450040123340576 + -0.32897580116872405*(-0.6740257057257448 + -0.0007917455797805673*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.5330540695702357*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((-0.6740257057257448 + -0.0007917455797805673*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.5330540695702357*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2 + 0.43090905819531766*(0.3709718635886192 + -0.47514710764751555*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.6627028336583534*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((0.3709718635886192 + -0.47514710764751555*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.6627028336583534*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((0.48450040123340576 + -0.32897580116872405*(-0.6740257057257448 + -0.0007917455797805673*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.5330540695702357*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((-0.6740257057257448 + -0.0007917455797805673*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.5330540695702357*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2 + 0.43090905819531766*(0.3709718635886192 + -0.47514710764751555*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.6627028336583534*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)*(1 + erf((0.3709718635886192 + -0.47514710764751555*(-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.08584163188401472 + 0.6683752749724832*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + -0.39230803394698555*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2 + 0.6627028336583534*(-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)*(1 + erf((-0.7769463282105753 + 0.25325339074048747*(0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))*(1 + erf((0.3679628458703208 + -0.7706264460934666*$(x[1]) + -0.03424767458492983*$(x[2]))/sqrt(2)))/2 + 0.5840452662855768*(0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))*(1 + erf((0.40575897127542815 + 0.13151803175485233*$(x[1]) + 0.5944153777408365*$(x[2]))/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2)/sqrt(2)))/2 - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    