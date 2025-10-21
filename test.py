
import torch
print(torch.version)
print(torch.version.cuda)
print(torch.cuda.is_available())

if torch.cuda.is_available():
  pass