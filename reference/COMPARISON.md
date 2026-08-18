# Comparação acadêmica: notebook do professor × implementação final

O notebook `PREPARANDO_DADOS_DO_DATAFRAME_TITANIC_PARA_K_NN.ipynb` é preservado sem alteração como material oficial de referência.

| Tema | Professor | Implementação final | Decisão e motivo |
|---|---|---|---|
| Fonte e caminho | Usa caminho absoluto de exemplo do Colab. | Usa `data/train.csv` relativo à raiz. | Permite reprodução local sem caminho absoluto. |
| Preparação | Demonstra Pandas no dataframe completo. | Divide treino/teste antes de ajustar mediana, moda e Min-Max. | Evita *data leakage* na avaliação de ML. |
| Colunas irrelevantes | Não remove explicitamente as colunas de identificação/texto. | Exclui `PassengerId`, `Name`, `Ticket` e `Cabin` da matriz do k-NN. | Strings e identificadores não têm distância matemática útil neste exercício. |
| `Age` e `Embarked` | Mediana e moda. | Mantém mediana e moda, ajustadas no treino. | Preserva a orientação didática e evita vazamento. |
| `Sex` e `Embarked` | `Sex` 0/1 e dummies de `Embarked`. | Mantém a mesma codificação; `Embarked` usa dummies com `drop_first=True`. | Alinhamento direto ao material. |
| `Pclass` | Inclui `Pclass` no Min-Max como número contínuo. | Usa dummies no experimento C. | A orientação textual mais recente define classe como categoria, não distância contínua. |
| Escala | Apresenta Min-Max manual para `Age`, `Fare` e `Pclass`. | Mantém a fórmula manual e escala `Age`, `Fare` e `FamilySize`. | `Pclass` passa a ser dummy; dummies já estão em 0/1. |
| K-NN e validação | Não treina nem avalia modelo. | Acrescenta A/B/C, seleção de `k` por CV com imputação e escala reajustadas dentro de cada fold, métricas, gráficos e verificação manual. | Completa a atividade sem substituir a demonstração de Pandas e evita leakage entre folds. |
