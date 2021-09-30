using JuMP, EAGO

                     m = Model()

                     register(m, :tsoftplus, 1, tsoftplus, autodiff = true)

                     @variable(m, -1 <= x[i=1:5] <= 1)

                     @variable(m, -31.132197238858367 <= q <= 26.880517306022778)

                     add_NL_constraint(m, :(tsoftplus(0.7425196467553086 + 0.75362497639758*tsoftplus(0.16779737433595132 + 0.20296934413172218*tsoftplus(0.2084912816234672 + -0.2303850665976186*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + 0.3979327859212889*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.3936336756093999*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + -0.6960567804469049*tsoftplus(-0.546047901538262 + -0.9077436645058139*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.9217358738595953*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.18556345853504919*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + -0.6753360286421448*tsoftplus(0.2643859493277727 + -0.4875873139593776*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.5790089118963042*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.6180629103252984*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5])))) + 0.32712010863709606*tsoftplus(0.23105897026567535 + 0.7864782673109372*tsoftplus(0.2084912816234672 + -0.2303850665976186*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + 0.3979327859212889*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.3936336756093999*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + -0.3773605615003972*tsoftplus(-0.546047901538262 + -0.9077436645058139*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.9217358738595953*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.18556345853504919*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + 0.4108493448439825*tsoftplus(0.2643859493277727 + -0.4875873139593776*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.5790089118963042*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.6180629103252984*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5])))) + -0.5610867427839104*tsoftplus(-0.8917979675370962 + -0.8866932464195947*tsoftplus(0.2084912816234672 + -0.2303850665976186*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + 0.3979327859212889*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.3936336756093999*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + 0.29205439914021003*tsoftplus(-0.546047901538262 + -0.9077436645058139*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.9217358738595953*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.18556345853504919*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + -0.3055057591822439*tsoftplus(0.2643859493277727 + -0.4875873139593776*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.5790089118963042*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.6180629103252984*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))))) + tsoftplus(-0.8907597983183071 + 0.20852757037282288*tsoftplus(0.16779737433595132 + 0.20296934413172218*tsoftplus(0.2084912816234672 + -0.2303850665976186*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + 0.3979327859212889*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.3936336756093999*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + -0.6960567804469049*tsoftplus(-0.546047901538262 + -0.9077436645058139*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.9217358738595953*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.18556345853504919*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + -0.6753360286421448*tsoftplus(0.2643859493277727 + -0.4875873139593776*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.5790089118963042*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.6180629103252984*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5])))) + -0.45428488438794323*tsoftplus(0.23105897026567535 + 0.7864782673109372*tsoftplus(0.2084912816234672 + -0.2303850665976186*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + 0.3979327859212889*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.3936336756093999*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + -0.3773605615003972*tsoftplus(-0.546047901538262 + -0.9077436645058139*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.9217358738595953*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.18556345853504919*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + 0.4108493448439825*tsoftplus(0.2643859493277727 + -0.4875873139593776*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.5790089118963042*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.6180629103252984*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5])))) + 0.17572288930791302*tsoftplus(-0.8917979675370962 + -0.8866932464195947*tsoftplus(0.2084912816234672 + -0.2303850665976186*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + 0.3979327859212889*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.3936336756093999*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + 0.29205439914021003*tsoftplus(-0.546047901538262 + -0.9077436645058139*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.9217358738595953*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.18556345853504919*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + -0.3055057591822439*tsoftplus(0.2643859493277727 + -0.4875873139593776*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.5790089118963042*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.6180629103252984*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))))) + tsoftplus(-0.568209606089721 + -0.7349333860588856*tsoftplus(0.16779737433595132 + 0.20296934413172218*tsoftplus(0.2084912816234672 + -0.2303850665976186*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + 0.3979327859212889*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.3936336756093999*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + -0.6960567804469049*tsoftplus(-0.546047901538262 + -0.9077436645058139*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.9217358738595953*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.18556345853504919*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + -0.6753360286421448*tsoftplus(0.2643859493277727 + -0.4875873139593776*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.5790089118963042*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.6180629103252984*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5])))) + -0.9973651442860416*tsoftplus(0.23105897026567535 + 0.7864782673109372*tsoftplus(0.2084912816234672 + -0.2303850665976186*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + 0.3979327859212889*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.3936336756093999*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + -0.3773605615003972*tsoftplus(-0.546047901538262 + -0.9077436645058139*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.9217358738595953*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.18556345853504919*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + 0.4108493448439825*tsoftplus(0.2643859493277727 + -0.4875873139593776*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.5790089118963042*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.6180629103252984*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5])))) + -0.536918017086788*tsoftplus(-0.8917979675370962 + -0.8866932464195947*tsoftplus(0.2084912816234672 + -0.2303850665976186*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + 0.3979327859212889*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.3936336756093999*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + 0.29205439914021003*tsoftplus(-0.546047901538262 + -0.9077436645058139*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.9217358738595953*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.18556345853504919*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))) + -0.3055057591822439*tsoftplus(0.2643859493277727 + -0.4875873139593776*tsoftplus(-0.4945840572579199 + -0.43859325607015887*$(x[1]) + -0.23075153863324926*$(x[2]) + 0.6886623503715517*$(x[3]) + 0.2530437009549349*$(x[4]) + -0.5381281144617338*$(x[5])) + -0.5790089118963042*tsoftplus(-0.2624361419722687 + 0.927050844450096*$(x[1]) + -0.8225783426220241*$(x[2]) + -0.22694600946869636*$(x[3]) + -0.42713541044759484*$(x[4]) + 0.09677754958817175*$(x[5])) + 0.6180629103252984*tsoftplus(0.6895569276686757 + -0.8200091221694046*$(x[1]) + -0.33926806377588514*$(x[2]) + 0.0373716517377054*$(x[3]) + 0.9771001762694858*$(x[4]) + -0.6175510698550335*$(x[5]))))) - $q <= 0.0))

                     @objective(m, Min, q)

                     return m

                    