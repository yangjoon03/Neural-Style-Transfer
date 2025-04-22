import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import copy

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 로딩 함수
def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.CenterCrop(imsize),
        transforms.ToTensor()
    ])
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# 이미지 출력 함수
def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.001)

# Content Loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

# Gram Matrix 계산
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

# Style Loss
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# 정규화 레이어
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, x):
        return (x - self.mean) / self.std

# 모델 구성 함수
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                style_img, content_img):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:i+1]

    return model, style_losses, content_losses

# 설정
imsize = 512 if torch.cuda.is_available() else 128
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# 이미지 불러오기
style_img = image_loader("style.jpg", imsize)
content_img = image_loader("content.jpg", imsize)
input_img = content_img.clone()

# VGG19 모델 불러오기
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# 모델 생성
model, style_losses, content_losses = get_style_model_and_losses(
    cnn, cnn_normalization_mean, cnn_normalization_std,
    style_img, content_img
)

# 옵티마이저
optimizer = optim.LBFGS([input_img.requires_grad_()])

# 스타일 트랜스퍼 수행
num_steps = 390
style_weight = 1e7
content_weight = 200

print("Optimizing...")
run = [0]

# GIF 프레임 저장용 리스트
frames = []

# 첫 프레임에 원본 content 이미지 추가
img = content_img.clone().detach().cpu().squeeze(0)
img = transforms.ToPILImage()(img)
frames.append(img)

while run[0] <= num_steps:
    def closure():
        input_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)

        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        loss = style_score * style_weight + content_score * content_weight
        loss.backward()

        run[0] += 1
        if run[0] % 10 == 0:
            print(f"Step {run[0]}:")
            print(f"Style Loss: {style_score.item():4f}, Content Loss: {content_score.item():4f}")

            # 현재 결과 이미지 표시
            imshow(input_img, title=f"Step {run[0]}")

            # 프레임 저장
            img = input_img.clone().detach().cpu().squeeze(0)
            img = transforms.ToPILImage()(img)
            frames.append(img)

        return loss

    optimizer.step(closure)

# 최종 결과 clamp
input_img.data.clamp_(0, 1)

# 최종 결과 표시
imshow(input_img, title="Final Output")
plt.show()


