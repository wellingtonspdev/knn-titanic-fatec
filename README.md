# K-NN Titanic — FATEC

Atividade de Aprendizagem de Máquina da aula de 18-08-2026. O notebook mostra, passo a passo, como preparar o Titanic e usar k-Nearest Neighbors (k-NN) para prever `Survived`.

## Objetivo e k-NN

Cada passageiro é um vetor de atributos. Para prever uma classe, o k-NN calcula a distância Euclidiana até os vetores de treino, seleciona os `k` vizinhos mais próximos e decide por votação majoritária. Como a distância é o centro do algoritmo, os dados precisam ser numéricos, sem valores ausentes e comparáveis em escala.

## Tecnologias

- Python 3.13+
- Pandas, scikit-learn e Matplotlib
- Jupyter Notebook
- pytest

## Estrutura

```text
knn-titanic-fatec/
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── gender_submission.csv
│   └── README.md
├── reference/
│   ├── PREPARANDO_DADOS_DO_DATAFRAME_TITANIC_PARA_K_NN.ipynb
│   └── COMPARISON.md
├── notebooks/
│   └── knn_titanic.ipynb
├── src/preprocessing.py
├── tests/test_preprocessing.py
├── submission_knn.csv
├── requirements.txt
└── README.md
```

## Materiais e papéis dos datasets

Os CSVs são os materiais oficiais da competição **Titanic — Kaggle**. Eles são locais e ignorados pelo Git; este projeto não declara autoria nem licença sobre esses dados.

As instruções exatas para obtê-los e posicioná-los estão em `data/README.md`.

- `data/train.csv` (891 linhas, 12 colunas): única base para exploração, preparação, divisão interna treino/teste, escolha de `k` e métricas A/B/C.
- `data/test.csv` (418 linhas, 11 colunas): teste externo sem `Survived`; é usado somente na rotina opcional de submissão, jamais para acurácia ou hiperparâmetros.
- `data/gender_submission.csv` (418 linhas, 2 colunas): exemplo do schema `PassengerId,Survived`; não é gabarito do `test.csv` e não participa de cálculo de acurácia.
- `reference/PREPARANDO_DADOS_DO_DATAFRAME_TITANIC_PARA_K_NN.ipynb`: notebook oficial do professor, preservado como referência. A comparação acadêmica está em `reference/COMPARISON.md`.

## Ambiente e execução

No PowerShell, na raiz:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter notebook notebooks/knn_titanic.ipynb
```

Abra o notebook e use **Restart Kernel → Run All**. Os caminhos são relativos ao projeto, sem dependência de caminhos absolutos do Windows.

Para executar os testes:

```powershell
python -m pytest -q
```

## Cinco etapas de preparação

1. **Feature selection:** remove `PassengerId`, `Name`, `Ticket` e `Cabin`; `Survived` permanece somente como alvo.
2. **Imputação:** `Age` recebe a mediana e `Embarked` recebe a moda, ambas ajustadas exclusivamente no treino. Na submissão externa, o único `Fare` nulo do `test.csv` recebe a mediana aprendida no `train.csv`.
3. **Encoding:** `Sex` é `male=0` e `female=1`; `Embarked` usa `pd.get_dummies(..., drop_first=True)`.
4. **Feature engineering:** `FamilySize = SibSp + Parch + 1`; no experimento C ela substitui `SibSp` e `Parch`.
5. **Escala:** Min-Max manual em `Age`, `Fare` e `FamilySize`; dummies e `Sex` já estão em 0/1.

No experimento C, `Pclass` vira dummies. Embora apareça como 1, 2 e 3, ela representa categorias de classe e não uma distância contínua apropriada.

### Normalização manual e leakage

O notebook mostra explicitamente:

```text
x_norm = (x - x_min) / (x_max - x_min)
```

`Fare` tem amplitude muito maior que `Pclass`, portanto poderia dominar a distância sem escala. A divisão é `test_size=0.20`, `random_state=42`, `stratify=y`. Mediana, moda, mínimo e máximo são aprendidos apenas no treino e reutilizados no teste interno; não são recalculados com o teste.

## Escolha de k e resultados reais

Os valores ímpares `1, 3, 5, 7, 9, 11, 13 e 15` são avaliados por validação cruzada estratificada exclusivamente no treino. Dentro de cada fold, mediana, moda, mínimo e máximo são ajustados apenas no fold-treino e reutilizados no fold-validação. `k=1` pode reagir a ruído, enquanto um `k` muito grande suaviza excessivamente as fronteiras; valores ímpares reduzem empates binários. O teste interno só é usado na avaliação final.

| Experimento | Features | k escolhido | Acurácia teste | Matriz de confusão |
|---|---|---:|---:|---|
| A | `Age`, `Fare` | 15 | 60,89% | `[[84, 26], [44, 25]]` |
| B | `Age`, `Fare`, `Sex` | 15 | 76,54% | `[[91, 19], [23, 46]]` |
| C | `Age`, `Fare`, `Sex`, `FamilySize`, `Pclass_1/2/3`, `Embarked_Q/S` | 15 | 78,21% | `[[100, 10], [29, 40]]` |

As linhas das matrizes são classes reais 0/1 e as colunas são previsões 0/1. A inclusão de `Sex` elevou B em **15,64 pontos percentuais** sobre A. O modelo expandido C elevou a acurácia em **17,32 p.p.** sobre A e **1,68 p.p.** sobre B. Esses valores foram gerados pelo notebook com o `train.csv` local oficial, não foram codificados manualmente.

O notebook também contém gráficos de acurácia de validação por `k` e comparação A/B/C. Para `PassengerId=566`, ele demonstra manualmente o modelo C corrigido (`k=15`) e reproduz separadamente a checagem originalmente declarada com `k=3`: os vizinhos `70`, `393` e `425` votam `0`, e a previsão manual coincide com a do scikit-learn.

## Submissão Kaggle opcional

Após avaliar A/B/C, a última célula treina C com todo o `train.csv` e gera `submission_knn.csv` com 418 linhas no schema exigido:

```text
PassengerId,Survived
892,0
893,0
```

O arquivo foi apenas gerado localmente; nenhum upload ao Kaggle é realizado.

## Testes e conclusões

Os testes cobrem Min-Max manual, parâmetros do treino reutilizados no teste, `FamilySize`, encoding, ausência de NaN, matriz numérica, remoção de `Survived` e preparação do `test.csv` externo sem alvo. A execução validada obteve **8 testes aprovados**.

As conclusões práticas são que k-NN não aceita strings/NaN, a escala altera diretamente as distâncias, categorias numéricas nem sempre são contínuas e a escolha de `k` deve ocorrer sem consultar repetidamente o conjunto de teste.
