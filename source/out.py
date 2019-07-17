import torch
from torchsummary import summary
from models import ResidualStackedHourGlass
from models import StackedHourGlass
from models import DeepPose
from models import ChainedPredictions
import sys

ORG_STDOUT = sys.stdout

f = open("ResidualStackedHourGlassSummary.txt", 'w')
sys.stdout = f

device = torch.device("cuda")
model = ResidualStackedHourGlass(256, 5, 2, 4, 16).to(device)
summary(model, (3, 256, 256))
f.close()

f = open("StackedHourGlassSummary.txt", 'w')
sys.stdout = f
model = StackedHourGlass(256, 2, 2, 4, 16).to(device)
summary(model, (3, 256, 256))
f.close()

f = open("ChainedPredictionsSummary.txt", 'w')
sys.stdout = f
model = ChainedPredictions("resnet50", 1, 1, 16).to(device)
summary(model, (3, 256, 256))
f.close()

f = open("DeepPoseSummary.txt", 'w')
sys.stdout = f
model = DeepPose(16, "resnet34").to(device)
summary(model, (3, 256, 256))
f.close()

sys.stdout = ORG_STDOUT
