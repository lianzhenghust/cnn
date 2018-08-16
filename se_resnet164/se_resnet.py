import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import math
from torch.autograd import Variable
import torch.nn.init as init



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)




class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(self.expansion*planes, self.expansion*planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(self.expansion*planes // 16, self.expansion*planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        # Excitation
        out = out * w
        out += shortcut
        return out


class SeResNet(nn.Module):
    def __init__(self, block, num_blocks, filters, num_classes=100):
        super(SeResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = conv3x3(3, self.in_planes)
        self.stage1 = self._make_layer(block, filters[0], num_blocks[0], stride=1)
        self.stage2 = self._make_layer(block, filters[1], num_blocks[1], stride=2)
        self.stage3 = self._make_layer(block, filters[2], num_blocks[2], stride=2)
        self.bn = nn.BatchNorm2d(filters[2]*block.expansion)
        self.linear = nn.Linear(filters[2]*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SeResNet164(num_classes=100):
    return SeResNet(PreActBottleneck, [18, 18, 18], [16, 32, 64], num_classes)




  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True 

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
# parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")  #恢复训练时的模型路径
# args = parser.parse_args()

# 超参数设置
EPOCH = 150   #遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 64      #批处理尺寸(batch_size)
LR = 0.001        #学习率

# 准备数据集并预处理
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
#     transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                     transforms.RandomCrop(32, padding=4),
                     transforms.ToTensor(),
                     transforms.Normalize((n/255.
                        					for n in [129.3, 124.1, 112.4]), (n/255. for n in [68.2,  65.4,  70.4]))])

transform_test = transforms.Compose([transforms.ToTensor(),
                     				transforms.Normalize((n/255.
                        								for n in [129.3, 124.1, 112.4]), (n/255. for n in [68.2,  65.4,  70.4]))])



trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)



# 模型定义-ResNet
net = SeResNet164().to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.Adam(net.parameters(), lr=LR) #, weight_decay=5e-4优化方式为mini-batcm，并采用L2正则化（权重衰减）


# 训练
with open("./data/acc.txt", "w") as f:
    with open("./data/log.txt", "w") as f2:
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
                outputs = net(inputs)
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
