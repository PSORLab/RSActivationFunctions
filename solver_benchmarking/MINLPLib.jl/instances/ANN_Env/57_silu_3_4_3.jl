using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:3] <= 1)

                     @variable(m, -12.331138821460632 <= q <= 11.287379993959185)

                     add_NL_constraint(m, :(swish(0.16893740747119512 + 0.361142465903193*swish(0.3759145594542668 + 0.23452173305395663*swish(-0.8645963938709578 + -0.1467408759040536*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.7957413591360925*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + 0.6670298352855966*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + 0.192500882174786*swish(-0.9609056399980664 + 0.06060702618594416*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + -0.49474851412125664*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.5046120388520428*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.6554806276835787*swish(-0.9973666264873402 + 0.1126936245130743*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.2338493567409592*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.8532702233572196*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3])))) + 0.21914525995597334*swish(-0.017859443401365915 + -0.7776638835685663*swish(-0.8645963938709578 + -0.1467408759040536*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.7957413591360925*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + 0.6670298352855966*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.3390165730930925*swish(-0.9609056399980664 + 0.06060702618594416*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + -0.49474851412125664*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.5046120388520428*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.16349638009140932*swish(-0.9973666264873402 + 0.1126936245130743*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.2338493567409592*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.8532702233572196*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3])))) + 0.7287835676612944*swish(0.6615910141517634 + 0.18731902906517428*swish(-0.8645963938709578 + -0.1467408759040536*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.7957413591360925*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + 0.6670298352855966*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.853933007586007*swish(-0.9609056399980664 + 0.06060702618594416*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + -0.49474851412125664*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.5046120388520428*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.9912964912419922*swish(-0.9973666264873402 + 0.1126936245130743*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.2338493567409592*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.8532702233572196*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))))) + swish(-0.9123133183860701 + 0.273513749523993*swish(0.3759145594542668 + 0.23452173305395663*swish(-0.8645963938709578 + -0.1467408759040536*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.7957413591360925*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + 0.6670298352855966*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + 0.192500882174786*swish(-0.9609056399980664 + 0.06060702618594416*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + -0.49474851412125664*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.5046120388520428*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.6554806276835787*swish(-0.9973666264873402 + 0.1126936245130743*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.2338493567409592*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.8532702233572196*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3])))) + 0.052599029592083735*swish(-0.017859443401365915 + -0.7776638835685663*swish(-0.8645963938709578 + -0.1467408759040536*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.7957413591360925*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + 0.6670298352855966*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.3390165730930925*swish(-0.9609056399980664 + 0.06060702618594416*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + -0.49474851412125664*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.5046120388520428*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.16349638009140932*swish(-0.9973666264873402 + 0.1126936245130743*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.2338493567409592*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.8532702233572196*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3])))) + 0.2642412570560455*swish(0.6615910141517634 + 0.18731902906517428*swish(-0.8645963938709578 + -0.1467408759040536*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.7957413591360925*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + 0.6670298352855966*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.853933007586007*swish(-0.9609056399980664 + 0.06060702618594416*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + -0.49474851412125664*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.5046120388520428*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.9912964912419922*swish(-0.9973666264873402 + 0.1126936245130743*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.2338493567409592*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.8532702233572196*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))))) + swish(0.5514504616591211 + 0.27732859795053555*swish(0.3759145594542668 + 0.23452173305395663*swish(-0.8645963938709578 + -0.1467408759040536*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.7957413591360925*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + 0.6670298352855966*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + 0.192500882174786*swish(-0.9609056399980664 + 0.06060702618594416*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + -0.49474851412125664*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.5046120388520428*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.6554806276835787*swish(-0.9973666264873402 + 0.1126936245130743*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.2338493567409592*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.8532702233572196*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3])))) + -0.08702067567597149*swish(-0.017859443401365915 + -0.7776638835685663*swish(-0.8645963938709578 + -0.1467408759040536*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.7957413591360925*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + 0.6670298352855966*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.3390165730930925*swish(-0.9609056399980664 + 0.06060702618594416*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + -0.49474851412125664*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.5046120388520428*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.16349638009140932*swish(-0.9973666264873402 + 0.1126936245130743*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.2338493567409592*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.8532702233572196*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3])))) + 0.40713382739201665*swish(0.6615910141517634 + 0.18731902906517428*swish(-0.8645963938709578 + -0.1467408759040536*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.7957413591360925*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + 0.6670298352855966*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.853933007586007*swish(-0.9609056399980664 + 0.06060702618594416*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + -0.49474851412125664*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.5046120388520428*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))) + -0.9912964912419922*swish(-0.9973666264873402 + 0.1126936245130743*swish(0.5833709147085329 + -0.6819730710434229*$(x[1]) + 0.5963453169666697*$(x[2]) + -0.1905949830790421*$(x[3])) + 0.2338493567409592*swish(0.2875838686662693 + -0.4272251449908575*$(x[1]) + 0.9704596097299736*$(x[2]) + -0.7281111839596055*$(x[3])) + -0.8532702233572196*swish(0.428536919496763 + 0.9920670296914205*$(x[1]) + 0.8818625631698058*$(x[2]) + -0.43136616052165655*$(x[3]))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    