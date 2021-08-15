# yen, wp

TOPIC: Supergat
URL: https://openreview.net/pdf?id=Wi5KUNlqWty
간략 설명: "4 가지의 방법을 비교: GO, DP, SD, MX
GO: GAT Original Single Layer Neural Network
(concat 후 a라는 vector 써서 곱하는 방식)
DP: Dot-Product
(그냥 feature a, b를 dot product로 dot(a, b.T) 하는 방식)
SD: Scaled Dot-Product
(DP와 같지만 scale 해서 sqrt(F)를 나누는 방식)
MX: GO와 DP를 mix 하는 방식
비교 결과, Node Classification의 경우 큰 차이는 없지만 DP가 GO보다 대부분 더 좋았음.
MX가 SD보다 Cora, CiteSeer, PubMed 에서 모두 더 좋은 성능을 보임
질문: 원래 GAT에서 Label을 고려했었는가? "
코드: https://github.com/dongkwan-kim/SuperGAT