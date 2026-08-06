## 수렴이 빠른 Muon과 일반화 성능이 좋은 SAM을 결합한 새로운 Optimizer

Muon의 고질적인 문제인 lr fine tuning문제를 해결하기 위해 AdaMuon을 채택했다.

<img width="579" height="247" alt="화면 캡처 2026-07-28 105139" src="https://github.com/user-attachments/assets/e74a1697-119c-4190-93f1-f4f42c65d955" />


코드의 G값을 Adaptive perturbation을 적용하고 계산했다(SAM의 insight).

모든 실험의 SAM은 Adaptive SAM(ASAM)을 사용했다.

SAM-in-Late-Phase 논문을 benchmark 기준으로 하여 새로운 Optimizer를 구현했다.

github link: https://github.com/zzp1012/SAM-in-Late-Phase.git

<img width="395" height="248" alt="화면 캡처 2026-07-28 105350" src="https://github.com/user-attachments/assets/b7393bbd-fc47-4d8f-9dd5-28d2114729dd" />
