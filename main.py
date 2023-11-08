import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras_core as keras

# teste gpu with torch
print(torch.cuda.is_available())
# git change branch name
# git branch -m <oldname> <newname>