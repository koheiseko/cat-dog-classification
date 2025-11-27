import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

def tensor_to_pil(tensor, mean, std):

    tensor = tensor.clone().cpu()

    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    
    tensor = tensor * std_tensor + mean_tensor
    
    tensor = torch.clamp(tensor, 0, 1)
    
    converter = T.ToPILImage()
    image_pil = converter(tensor)
    
    return image_pil

def evaluation(model, data_loader, metric, device, mean, std):
    classes = {0: "gato", 1: "cachorro"}
    model.eval()
    
    with torch.no_grad():
        for X, y_true in data_loader:
            X, y_true = X.to(device), y_true.to(device)
            y_pred = model(X)
            prob = F.softmax(y_pred)
            classe = torch.argmax(prob, dim=1).item()
            
            X = X.reshape(3, 224, 224)

            img_pil = tensor_to_pil(X, mean, std)

            display(img_pil) 
            print(f"O animal na imagem é um {classes[classe]}")

            metric.update(y_pred, y_true)
    return metric.compute().item()
