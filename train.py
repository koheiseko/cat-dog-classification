import torch
import torch.nn as nn
import torchmetrics
from src.engine import train, evaluate
from src.data_setup import get_train_data_transformed, get_test_data_transformed, get_data_loader, train_test_split, load_dataset
from src.model import ResNetTransfer
import os

SEED = 42
IMAGE_SIZE = 224
LEARNING_RATE = 0.0001
N_EPOCHS = 3
N_EPOCHS_FINE_TUNING = 2
N_CLASSES = 2
BATCH_SIZE = 32
TRAINING_SIZE = 0.2


def main():

    os.makedirs(name="data", exist_ok=True)

    print("[INFO] Iniciando o download dos dados")
    dataset_name = "tongpython/cat-and-dog"
    load_dataset(dataset_name, "data")
    print("[INFO] Download concluído!")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed(SEED)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_and_valid_data = get_train_data_transformed("data/training_set", IMAGE_SIZE, 0.5, 10, imagenet_mean, imagenet_std)
    test_data = get_test_data_transformed("data/test_set", IMAGE_SIZE, imagenet_mean, imagenet_std)
    train_data, valid_data = train_test_split(train_and_valid_data, TRAINING_SIZE)

    train_loader, valid_loader, test_loader = get_data_loader(train_data, valid_data, test_data, BATCH_SIZE, True)

    print("[INFO] Inicializando o modelo")
    model = ResNetTransfer(n_classes=N_CLASSES)
    model = model.to(device)

    xentropy = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=N_CLASSES).to(device)

    print(f"[INFO] Iniciando o treinamento do modelo por {N_EPOCHS} épocas")
    history = train(model, optimizer, xentropy, accuracy, train_loader, valid_loader, N_EPOCHS, device)

    print(f"[INFO] Iniciando o processo de avaliação do modelo")
    model_acc = evaluate(model, test_loader, accuracy, device).compute().item()
    print(f"[RESULTADO] Acurácia: {model_acc * 100:.3f}%")

    os.makedirs(name="models", exist_ok=True)
    torch.save(model.state_dict(), "models/resnet_frozen.pth")
    print("[INFO] O modelo foi salvo com sucesso")

    print("[INFO] Iniciando o fine tuning do modelo")
    for param in model.parameters():
        param.requires_grad = True

    learning_rate_fine_tune = LEARNING_RATE / 1000

    optimizer_fine_tune = torch.optim.AdamW(params=model.parameters(), lr=learning_rate_fine_tune)

    history_fine_tune = train(model, optimizer_fine_tune, xentropy, accuracy, train_loader, valid_loader, N_EPOCHS_FINE_TUNING, device)

    print(f"[INFO] Iniciando o processo de avaliação do modelo após o fine tuning")
    model_fine_tuning_acc = evaluate(model, test_loader, accuracy, device).compute().item()
    print(f"[RESULTADO] Acurácia: {model_fine_tuning_acc * 100:.3f}%")

    torch.save(model.state_dict(), "models/resnet_finetuned.pth")
    print("[INFO] O modelo após o fine tuning foi salvo com sucesso")

if __name__ == "__main__":
    main()



