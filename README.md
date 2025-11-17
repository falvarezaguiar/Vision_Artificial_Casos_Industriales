# Inspección Visual Industrial con CNN

Replicación del experimento de Ingo Nowitzky para clasificar componentes electrónicos (bobinas con pines metálicos) como buenos o defectuosos.

**Artículo:** [Building a Vision Inspection CNN for an Industrial Application](https://medium.com/data-science/building-a-vision-inspection-cnn-for-an-industrial-application-138936d7a34a)  
**Autor:** Ingo Nowitzky (Nov 2024)

---

## Objetivo

Clasificar componentes electrónicos (bobinas con pines metálicos) como:
- **OK (1_io):** Pines correctamente alineados
- **NOK (0_nio):** Pines doblados/defectuosos

## Dataset

- **Total:** 624 imágenes BMP en escala de grises
- **Distribución:**
  - OK (1_io): 500 imágenes (80%)
  - NOK (0_nio): 124 imágenes (20%)
- **Dimensiones:** 400x700 px
- **Split (según artículo sección 2.4):** 50% train / 30% val / 20% test
  - Train: ~312 imágenes (50%)
  - Val: ~187 imágenes (30%)
  - Test: ~125 imágenes (20%)

## Estructura del Proyecto

```
PF1-VA/
├── data/
│   └── Coil_Vision/
│       ├── 01_train_val_test/  # Dataset original (624 imágenes)
│       ├── train/              # Split 50% entrenamiento
│       ├── val/                # Split 30% validación
│       ├── test/               # Split 20% test
│       └── 02_predict/         # Imágenes para predicción
├── src/
│   ├── model.py                # Arquitectura CNN
│   ├── data_utils.py           # Split del dataset (50/30/20)
│   ├── train.py                # Entrenamiento
│   ├── predict.py              # Inferencia
│   └── gradcam.py              # Visualización GradCAM
├── notebooks/
│   └── vision_inspection_colab.ipynb  # Notebook completo para Colab
├── models/                     # Modelos entrenados (.pth)
├── resultados/                 # Gráficas y resultados
└── requirements.txt
```

## Arquitectura del Modelo

```
Input: 1x400x700 (grayscale)
├── Conv2d(1→6, kernel=5x5) → ReLU → MaxPool(2x2)
├── Conv2d(6→16, kernel=5x5) → ReLU → MaxPool(2x2)
├── Flatten (266,944 valores)
├── Linear(266944→120) → ReLU
└── Linear(120→2) [NOK, OK]

Total: 32,036,214 parámetros
```

## Hiperparámetros

- **Épocas:** 60 (notebook) / 30 (default en train.py)
- **Batch size:** 4
- **Learning rate:** 0.001
- **Optimizer:** SGD (notebook) / Adam (train.py - pendiente actualizar)
- **Loss:** CrossEntropyLoss
- **Balanceo:** WeightedRandomSampler para balancear clases en training

## Uso

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Dividir dataset en train/val/test (50/30/20)

```bash
python src/data_utils.py
```

Esto creará los directorios `train/`, `val/` y `test/` con la distribución correcta.

### 3. Entrenar modelo (local - CPU)

```bash
python src/train.py
```

**Nota:** El entrenamiento en CPU puede tardar varias horas. Se recomienda usar GPU.

### 4. Entrenar en Colab (GPU recomendado)

1. Subir el proyecto a Google Drive
2. Abrir `notebooks/vision_inspection_colab.ipynb` en Google Colab
3. Configurar runtime con GPU (Runtime → Change runtime type → GPU)
4. Ejecutar todas las celdas

**Tiempo estimado:** ~15 minutos con GPU T4

### 5. Realizar predicciones

```bash
python src/predict.py
```

Coloca las imágenes a predecir en `data/Coil_Vision/02_predict/`

### 6. Visualizar GradCAM

```bash
python src/gradcam.py
```

## Resultados

### Objetivo del Artículo
- **Validation Accuracy:** >95%

### Resultados Obtenidos (60 épocas)
- **Train Accuracy:** ~98-99%
- **Validation Accuracy:** ~95-97%
- **Test Accuracy:** ~99% (según notebook)
- **Observación:** Se detecta overfitting (gap entre train y val loss)

### Análisis de Overfitting

El modelo muestra signos de overfitting:
- Train loss → ~0.0 (época 60)
- Val loss → ~0.15-0.2 (se estabiliza)
- Gap significativo entre train y validation accuracy

**Causas probables:**
- Modelo grande (32M parámetros) para dataset pequeño (312 imágenes training)
- Sin regularización (dropout, weight decay)
- Sin data augmentation

## Próximos Pasos

1. **Implementar regularización:**
   - Agregar Dropout (0.3-0.5) en capas FC
   - Agregar weight decay al optimizer
   
2. **Data augmentation:**
   - Rotaciones pequeñas (±5°)
   - Flip horizontal
   - Aumentar variabilidad del dataset

3. **Early stopping:**
   - Detener entrenamiento cuando val loss deje de mejorar
   - Guardar mejor modelo según validation loss

4. **Visualización:**
   - Implementar GradCAM para visualizar decisiones del modelo
   - Analizar casos de error (falsos positivos/negativos)

5. **Arquitecturas alternativas:**
   - Probar modelos más simples
   - Transfer learning con modelos pre-entrenados

## Referencias

- **Artículo original:** [Building a Vision Inspection CNN for an Industrial Application](https://medium.com/data-science/building-a-vision-inspection-cnn-for-an-industrial-application-138936d7a34a)
- **Autor:** Ingo Nowitzky (Nov 2024)

## Notas Técnicas

- El notebook usa **SGD** como optimizer (según artículo)
- El código fuente (`src/train.py`) actualmente usa **Adam** (pendiente actualizar)
- El balanceo de clases se realiza con `WeightedRandomSampler` en el notebook
- El split 50/30/20 sigue exactamente la sección 2.4 del artículo original
