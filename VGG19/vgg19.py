import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn



cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

class VGG19(nn.Module):
	def __init__(self, cfg=cfg):
		super(VGG19, self).__init__()
		self.features = self._make_layers(cfg)
		self.classifier = nn.Linear(512, 10)

	def forward(self, x):
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return out

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
						   nn.BatchNorm2d(x),
						   nn.ReLU(inplace=True)]
				in_channels = x
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# 模型定义 VGG19
net = VGG19().to(device)


# 超参数设置
EPOCH = 50  #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 20      #批处理尺寸(batch_size)
LR = 0.001        #学习率

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data/', train=True, download=False, transform=transform_train) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='./data/', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)



# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
with open("/data/acc.txt", "w") as f:
    with open("/data/log.txt", "w") as f2:
        for epoch in range(pre_epoch, EPOCH):
            print('\nEpoch: %d' % (epoch + 1))
            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, data in enumerate(trainloader, 0):
                # 准备数据
                length = len(trainloader)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # forward + backward
                outputs = net(inputs).to(device)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                # 打印一次loss和准确率     
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d,  | loss: %.4f | acc: %.3f%% '
                  % (epoch + 1, sum_loss / (i + 1), 100 * float(correct) / total))
            f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                                  % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
            f2.write('\n')
            f2.flush()
            # 训练完测试一下准确率
            print("Waiting Test!")
            with torch.no_grad():
                correct = 0
                total = 0
                for data in testloader:
                    net.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images).to(device)
                    # 取得分最高的那个类 (outputs.data的索引号)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                print('测试分类准确率为：%.3f%%' % (100 * float(correct) / total))
                acc = 100 * float(correct) / total
                # 将每次测试结果实时写入acc.txt文件中
                print('Saving model......')
                f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                f.write('\n')
                f.flush()
        print("Training Finished, TotalEPOCH=%d" % EPOCH)


