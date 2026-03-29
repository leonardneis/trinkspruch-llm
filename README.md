# Trinkspruch LLM

Fine-tuning eines Language Models zur Generierung deutscher Trinksprüche mit QLoRA.

---

## Projektziel

Dieses Projekt trainiert ein kleines Language Model darauf, kreative, kurze und schlagfertige Trinksprüche zu erzeugen.

---

## Methodik

Das Modell wird mittels QLoRA feinjustiert.

### Verwendete Techniken

- 4-bit Quantisierung (bitsandbytes)
- LoRA (Low-Rank Adaptation)
- Instruction Tuning

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python training/train.py
```

## Inference

```bash
python inference/generate.py --prompt "Frecher Trinkspruch"
```

## Dataset

Der Datensatz besteht aus bereinigten und deduplizierten deutschen Trinksprüchen aus verschiedenen Quellen.

## Hinweise

- Modelle werden nicht im Repository gespeichert
- Training erfolgt mittels QLoRA zur Reduktion des Speicherverbrauchs

---

## Weiterführende Literatur

Dieses Projekt basiert auf etablierten Methoden zur effizienten Feinabstimmung großer Sprachmodelle:

---

- **QLoRA: Efficient Finetuning of Quantized LLMs**
  Dettmers et al., 2023
  https://arxiv.org/abs/2305.14314
  Beschreibt die Kombination aus 4-Bit-Quantisierung und LoRA für speichereffizientes Fine-Tuning.

---

- **LoRA: Low-Rank Adaptation of Large Language Models**
  Hu et al., 2021
  https://arxiv.org/abs/2106.09685
  Führt parameter-effizientes Fine-Tuning mittels Low-Rank-Matrizen ein.

---

- **Self-Instruct: Aligning Language Models with Self-Generated Instructions**
  Wang et al., 2023
  https://arxiv.org/abs/2212.10560
  Grundlage für Instruction-Tuning, wie es auch in diesem Projekt verwendet wird.

---

- **LLaMA: Open and Efficient Foundation Language Models**
  Meta AI, 2023
  https://arxiv.org/abs/2302.13971
  Liefert Kontext zu modernen Transformer-basierten Sprachmodellen.
