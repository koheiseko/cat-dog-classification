# Cats and Dogs Classifier - PyTorch ResNet18

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Status](https://img.shields.io/badge/Status-Training%20Complete-success)
![License](https://img.shields.io/badge/License-MIT-green)

Um projeto robusto de classificação de imagens end-to-end utilizando **Deep Learning** e **Transfer Learning**. O objetivo é distinguir entre imagens de gatos e cachorros com alta precisão, utilizando uma arquitetura **ResNet18** pré-treinada no ImageNet e aplicando técnicas avançadas de Fine-Tuning.

O projeto foi estruturado seguindo boas práticas de Engenharia de Machine Learning, com código modularizado, pipelines de dados reprodutíveis e separação clara entre lógica de treino e definição de modelo.

---

## Arquitetura e Estratégia

O modelo baseia-se na **ResNet18**, uma Rede Neural Convolucional (CNN) profunda com conexões residuais. A estratégia de treinamento foi dividida em duas fases para maximizar a acurácia e evitar o "Catastrophic Forgetting":

1.  **Fase 1: Feature Extraction**
    * O backbone da ResNet18 (pré-treinado no ImageNet) é mantido congelado (`requires_grad=False`).
    * Apenas a nova "cabeça" (camada Linear final) é treinada.
    * Objetivo: Adaptar a saída para 2 classes sem destruir os filtros de detecção de bordas/formas já aprendidos.

2.  **Fase 2: Fine-Tuning**
    * Todas as camadas são descongeladas.
    * O modelo é treinado com um **Learning Rate drasticamente reduzido** (100x menor).
    * Objetivo: Refinar os filtros profundos para diferenciar características específicas de pelos de cães e gatos.

---

## Estrutura do Projeto

```text
├── data/                   # Diretório de dados (gerado automaticamente)
├── models/                 # Checkpoints salvos (.pth)
├── notebooks/              # Jupyter Notebooks para experimentação
├── src/                    # Código fonte modular
│   ├── data_setup.py       # Pipeline de ETL (DataLoaders, Transforms)
│   ├── engine.py           # Loops de treino e avaliação (agnóstico ao modelo)
│   ├── model.py            # Definição da arquitetura (ResNetTransfer)
│   └── utils.py            # Funções auxiliares
├── train.py                # Script principal de orquestração do treino
├── pyproject.toml          # Gerenciamento de dependências
├── uv.lock                 # Lockfile (uv)
└── README.md               # Documentação do projeto
```

## Como executar

1. Clone o repositório

```
git clone https://github.com/koheiseko/cat-dog-classification
cd cat-dog-classification
```

2. Instale as dependências

```
uc sync
```

3. Execute o arquivo train.py

```
uv run train.py
```