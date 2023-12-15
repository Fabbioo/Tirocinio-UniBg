# Third-part imports
import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights

def preprocess(image: torch.Tensor, resize: tuple) -> torch.Tensor:
    """
    \nObiettivo: preprocessare le immagini su cui eseguire l'attacco FGSM.
    \nInput:
    \n  - image: immagine da preprocessare.
    \n  - resize: tupla contenente le dimensioni a cui fare il resize dell'immagine per il preprocessing.
    \nOutput:
    \n  - torch.Tensor: immagine preprocessata con shape (a, b, c, d).
    """
    
    image = torch.clamp(image, 0, 255).to(torch.uint8)
    image = torchvision.transforms.functional.resize(image, [resize[0], resize[1]])
    image = image.float() / 255.
    normalization = torchvision.transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
    image = normalization(image)
    image = image.unsqueeze(0)
    
    return image

def inference(model: torchvision.models, image: torch.Tensor) -> (int, str, float):
    """
    \nObiettivo: fare inferenza sull'immagine.
    \nInput:
    \n  - model: modello da usare per fare inferenza.
    \n  - image: immagine su cui fare inferenza.
    \nOutput:
    \n  - int: id della classe predetta.
    \n  - str: etichetta della classe predetta.
    \n  - float: confidenza della predizione.
    """
    
    output = model(image).squeeze(0).softmax(0) # output.shape = torch.Size([1000])
    
    class_id = output.argmax().item()
    class_name = ResNet50_Weights.IMAGENET1K_V2.meta['categories'][class_id]
    class_conf = output[class_id].item()
    
    return (class_id, class_name, class_conf)

def fgsm_attack(model: torchvision.models, image: torch.Tensor, epsilon: float, device: str) -> (torch.Tensor, torch.Tensor):
    """
    \nObiettivo: eseguire l'attacco FGSM.
    \nInput:
    \n  - model: modello da usare per fare inferenza.
    \n  - image: immagine su cui fare inferenza.
    \n  - epsilon: valore scelto per il parametro epsilon.
    \n  - device: device da utilizzare.
    \nOutput:
    \n  - torch.Tensor: tensore rappresentante l'immagine perturbata da postprocessare.
    \n  - torch.Tensor: tensore rappresentante il noise aggiunto all'immagine di input.
    """
    
    image.requires_grad = True
    target: int = inference(model, image)[0]
    target = torch.Tensor([target]).long().to(device)
    real_pred = model(image).to(device) # real_pred.shape = torch.Size([1, 1000])
    loss_fn = nn.CrossEntropyLoss() # Funzione di loss
    loss = loss_fn(real_pred, target)
    model.zero_grad()
    loss.backward()
    grad = image.grad.data
    grad_sign = grad.sign()
    noise = epsilon * grad_sign
    perturbed_image = image + noise       
    
    return (perturbed_image, noise)

def postprocess(image: torch.Tensor) -> torch.Tensor:
    """
    \nObiettivo: postprocessare le immagini su cui è stato eseguito l'attacco FGSM.
    \nInput:
    \n  - image: immagine da postprocessare.
    \nOutput:
    \n  - torch.Tensor: immagine postprocessata.
    """
    
    denormalization = torchvision.transforms.Normalize(
        mean = [-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std = [1/0.229, 1/0.224, 1/0.255]
    )
    image = denormalization(image)
    image = torch.clamp(image, 0, 1) 
    
    return image