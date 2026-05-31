# trabalho_aplicado1_ast

Pipeline reprodutível do Trabalho Aplicado 1 da disciplina **Análise de Séries Temporais - MAD485**.

O projeto modela a série mensal `pmc_volume_index` com modelos ARIMA/SARIMA, compara especificações concorrentes em uma janela de teste observada e gera previsões futuras com intervalos de incerteza.

## Dados

Entrada bruta principal:

```text
data/raw/dataset_pmc_selic_monthly.csv
```

Base usada no modelo univariado:

```text
data/processed/pmc_monthly_model_input.csv
```

Colunas usadas na modelagem:

```text
date, pmc_volume_index
```

## Notebook principal

```text
notebooks/01_trabalho1_ast_pmc_arima_sarima.ipynb
```

O notebook cobre:

1. carregamento e validação dos dados;
2. análise exploratória;
3. testes de estacionariedade;
4. partição treino-teste;
5. estimação de modelos ARIMA/SARIMA concorrentes;
6. avaliação preditiva no período de teste;
7. diagnóstico residual;
8. previsão futura.

## Script reprodutível

O pipeline também pode ser executado por script:

```powershell
python .\src\run_trabalho1_pipeline.py
```

As saídas são gravadas em `reports/`.

