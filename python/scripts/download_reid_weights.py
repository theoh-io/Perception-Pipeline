import os
import wget

file_path = os.path.dirname(os.path.realpath(__file__))
print(file_path)
os.chdir("../src/perceptionloomo/deep_sort/deep/checkpoint/")
print(os.getcwd())

url = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R50.pth"
# wget.download(url)
try:
    os.rename('market_agw_R50.pth', 'resnet50_AGWmarket.pth')
except:
    print("No renaming 1")
url = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_R50.pth"
# wget.download(url)
try:
    os.rename('duke_sbs_R50.pth', 'resnet50_SBSduke.pth')
except:
    print("No renaming 2")
url = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R50.pth"
wget.download(url)
try:
    os.rename('msmt_sbs_R50.pth', 'resnet50_SBSmsmt17.pth')
except:
    print("No renaming 3")