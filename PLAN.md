# nano-KAN: nanoGPT + KAN Hybrid

## Идея
Заменить MLP-блоки в GPT-2 на **HybridKANMLP** — параллельную комбинацию
классического MLP и B-spline KAN. Attention остаётся без изменений.

## Ответы на вопросы
| Вопрос | Ответ |
|--------|-------|
| KAN-вариант | B-spline KAN (оригинальная статья) |
| Стратегия гибрида | Parallel: `out = MLP(x) + KAN(x)` |
| Размер модели | Small: 256d, 6 layers, 4 heads (~10M params) |
| Датасет | Shakespeare (char-level текст, BPE-токены) |
| Пайплайн | Полный: model + train + generate |
| Токенизатор | tiktoken GPT-2 BPE (vocab 50257) |

## Архитектура

```
Input IDs
    │
    ▼
Token Embedding + Position Embedding   (vocab=50257, n_embd=256)
    │
    ▼
┌──────────────── Transformer Block ×6 ───────────────┐
│                                                       │
│   LayerNorm → CausalSelfAttention (4 heads, 256d)    │
│       │  + residual                                   │
│       ▼                                               │
│   LayerNorm → HybridKANMLP                            │
│       │         ├── MLP branch:  256→1024→256          │
│       │         └── KAN branch:  256→256 (B-spline)   │
│       │         out = mlp + kan                        │
│       │  + residual                                   │
└───────┴───────────────────────────────────────────────┘
    │
    ▼
LayerNorm → Linear (256→50257)  → logits
```

## B-spline KAN
- Каждое ребро графа — обучаемый B-сплайн на сетке
- `KANLinear(in_features, out_features, grid_size=5, spline_order=3)`
- Выход: `base_activation(x) @ base_weight + spline(x) @ spline_weight`
- Base activation: SiLU

## Конфиг модели
```python
n_embd      = 256
n_layer     = 6
n_head      = 4
block_size  = 256
dropout     = 0.1
vocab_size  = 50257   # tiktoken gpt2
bias        = False

# KAN
kan_grid_size    = 5
kan_spline_order = 3
```

## Файлы проекта
```
nano-kan/
├── PLAN.md              ← этот файл
├── kan.py               ← B-spline KAN layer (KANLinear)
├── model.py             ← GPT + HybridKANMLP
├── train.py             ← цикл обучения (single GPU)
├── generate.py          ← генерация текста
└── data/
    └── shakespeare/
        └── prepare.py   ← скачивание и токенизация
```

## Шаги реализации
1. ✍️ Записать план (этот файл)
2. Реализовать `kan.py` — KANLinear с B-сплайнами
3. Реализовать `model.py` — GPT с HybridKANMLP
4. Реализовать `train.py` — обучение
5. Реализовать `generate.py` — генерация
6. Реализовать `data/shakespeare/prepare.py` — подготовка данных
7. Тест: проверить импорты и размерности тензоров
