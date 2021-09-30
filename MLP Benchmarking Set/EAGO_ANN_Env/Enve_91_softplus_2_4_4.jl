using JuMP, EAGO

                     m = Model()

                     EAGO.register_eago_operators!(m)

                     @variable(m, -1 <= x[i=1:2] <= 1)

                     @variable(m, -16.465220391252963 <= q <= 13.247242945851657)

                     add_NL_constraint(m, :(softplus(-0.8607261300549842 + -0.20051051560602984*softplus(0.32617165974118256 + -0.06774704341275761*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.8207829095648775*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.2715620289650058*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.2095707637249471*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2])))) + 0.26087215928789487*softplus(-0.6407062227441411 + -0.9593278201481437*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.8315933410434453*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.3107427624049257*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.42065976660830096*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2])))) + 0.6545660836315959*softplus(-0.33615666125837107 + -0.6070907255689493*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.29752219198541896*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.5804695626203649*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.5287504950541346*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2])))) + -0.23324651233588023*softplus(0.24668743225395762 + -0.6772776163948353*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.6697591502921436*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.9869294969555291*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.927705920337516*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))))) + softplus(0.779819989742899 + 0.8496895984834958*softplus(0.32617165974118256 + -0.06774704341275761*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.8207829095648775*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.2715620289650058*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.2095707637249471*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2])))) + -0.02204872969761329*softplus(-0.6407062227441411 + -0.9593278201481437*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.8315933410434453*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.3107427624049257*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.42065976660830096*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2])))) + -0.5110587240916482*softplus(-0.33615666125837107 + -0.6070907255689493*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.29752219198541896*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.5804695626203649*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.5287504950541346*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2])))) + 0.18452897453753314*softplus(0.24668743225395762 + -0.6772776163948353*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.6697591502921436*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.9869294969555291*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.927705920337516*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))))) + softplus(0.613621358738742 + -0.5615208399233356*softplus(0.32617165974118256 + -0.06774704341275761*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.8207829095648775*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.2715620289650058*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.2095707637249471*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2])))) + 0.7974600337622593*softplus(-0.6407062227441411 + -0.9593278201481437*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.8315933410434453*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.3107427624049257*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.42065976660830096*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2])))) + 0.6056489819777799*softplus(-0.33615666125837107 + -0.6070907255689493*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.29752219198541896*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.5804695626203649*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.5287504950541346*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2])))) + 0.38142942028117055*softplus(0.24668743225395762 + -0.6772776163948353*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.6697591502921436*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.9869294969555291*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.927705920337516*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))))) + softplus(-0.760664149694743 + -0.311119381775256*softplus(0.32617165974118256 + -0.06774704341275761*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.8207829095648775*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.2715620289650058*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.2095707637249471*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2])))) + -0.21629159440432089*softplus(-0.6407062227441411 + -0.9593278201481437*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.8315933410434453*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.3107427624049257*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.42065976660830096*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2])))) + -0.27663498120832664*softplus(-0.33615666125837107 + -0.6070907255689493*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.29752219198541896*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.5804695626203649*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.5287504950541346*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2])))) + 0.021677564434742003*softplus(0.24668743225395762 + -0.6772776163948353*softplus(0.7070898430033767 + 0.2372096627500988*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.5147399373372079*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.5339003867762222*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.21113635689367882*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + 0.6697591502921436*softplus(0.671496694593106 + 0.292444405668296*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.33061093540487363*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.46155532601272675*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + -0.8446058043764437*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.9869294969555291*softplus(-0.34538259413627204 + 0.4461300697720958*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.39900427087161594*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + 0.5606730316385984*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5702029544191971*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))) + -0.927705920337516*softplus(-0.42476197721268605 + 0.3936041134565156*softplus(0.816839553962569 + 0.6481777423377117*$(x[1]) + 0.36073552437969214*$(x[2])) + 0.1376878092454441*softplus(0.4148504430229685 + -0.7369336500806245*$(x[1]) + -0.10093138139684799*$(x[2])) + -0.21877344170036528*softplus(0.868843582590237 + -0.30974098839308173*$(x[1]) + 0.5957165540460112*$(x[2])) + 0.5281699546086327*softplus(0.577898792363158 + -0.5512051830310165*$(x[1]) + 0.08996029803466721*$(x[2]))))) - $q <= 0.0))

                     @NLobjective(m, Min, q)

                     return m

                    