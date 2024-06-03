import torch

# Create custom index ranges for 5 folds
# fold_indices = [
#     list(range(0, 10000)),
#     list(range(10000, 20000)),
#     list(range(20000, 30000)),
#     list(range(30000, 40000)),
#     list(range(40000, 50000))
# ]
# print(len(fold_indices))

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
# print("Is CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA current device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available. Please check your installation.")

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_properties(0).total_memory)
print(f"Is cuda available: {torch.cuda.is_available()}")