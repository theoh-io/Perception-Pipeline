import os
import wget

file_path = os.path.dirname(os.path.realpath(__file__))
print(file_path)
os.chdir("../src/dlav22/deep_sort/deep/checkpoint/")
print(os.getcwd())

url = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R50.pth"
wget.download(url)
os.rename('market_agw_R50.pth', 'resnet50_AGWmarket.pth')
url = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_R50.pth"
wget.download(url)
os.rename('market_agw_R50.pth', 'resnet50_AGWmarket.pth')
url = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R50.pth"
wget.download(url)
os.rename('market_agw_R50.pth', 'resnet50_AGWmarket.pth')
