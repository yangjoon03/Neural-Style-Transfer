# Neural Style Transfer


## 요약
이미지 화풍을 입히는 방법

## 네트워크 구조
![image](https://github.com/user-attachments/assets/f97e1d02-b34a-4fc4-ada4-f1d90ac5d269)
* inpaut size : 512 512
* output size : 512 512
* VGG19 모델의 백본을 사용하여 이미지의 특징을 추출
  1. content 정보는 고수준 특징 Deep Layer
  2. 스타일 정보는 Shallow Layer
* 
* 필요 이미지 : 2장
  1. Content 이미지
  2. Style 이미지


## 결과
1. weight
* style weight : 1e7
* content weight : 200
* 보편적으로 좋은 결과를 나타내었음

2. style 이미지
* 선이 강조되거나 뚜렷한 특징이 있는 이미지에서 좋은 결과를 나타냄
* content색과 style색이 비슷한 경우 더욱 좋은 결과를 보여주었음.
