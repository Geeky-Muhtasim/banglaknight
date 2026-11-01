# BanglaKnight data analysis: train, dev and devtest

Author: Muhtasim Ibteda Shochcho (analysis run with Claude) | Date: 2026-11-01 | Status: findings, not yet acted on

Scope: `train.jsonl` (8,196 prompts), `dev.jsonl` (2,049 prompts), `devtest.jsonl` (5,000 blind prompts), 3 forms per prompt. Only text and train/dev labels were used. No devtest labels exist and none were inferred by hand. Keyword groups are regex heuristics on the romanized form, so the shares below are approximate. "TF-IDF CV" means a 5-fold cross-validated logistic regression on romanized word n-grams plus Bangla character n-grams over train+dev (accuracy 0.750).

---

## 1. Summary

1. **Dev is much easier than devtest.** About 20% of train is a product-complaint template that is 96% Safe and almost perfectly classified. It is only 5% of devtest. A second subset (native Bangladeshi adversarial prompts, mostly Severe) is 6.6% of train and 1.9% of devtest. Devtest is 93% "hard core", where TF-IDF CV accuracy is 68% instead of 75%. This explains most of the 3 to 6.5 point dev-to-devtest drop, and why encoders (more template-driven) lose more than Qwen.
2. **Estimated devtest class mix**: Safe about 20 to 22%, Offensive 20 to 22%, Harmful 29 to 34%, Severe 27% (train: 34 / 15 / 26 / 24). Two independent estimates agree.
3. **The labels follow topic, not attack technique.** Prompt injection / "ignore the above" is 87% Harmful; Severe is driven by weapons, drugs, suicide, malware, child abuse and sexual violence.
4. **Labels are consistent.** Near-duplicates agree 97 to 100%; confident CV disagreements (1.7%) are mostly genuine Harmful/Severe boundary ambiguity, not errors.
5. **The code-mixed view carries generation artifacts** (Hindi presence correlates with Safe) and leaks words of the original English prompt.
6. **No leakage, no ordering effects, consistent romanization** across splits.

---

## 2. Basic statistics

| | train | dev | devtest |
|---|---|---|---|
| Prompts | 8,196 | 2,049 | 5,000 |
| Safe / Offensive / Harmful / Severe | 34.1 / 15.3 / 26.3 / 24.3 % | 34.1 / 15.3 / 26.3 / 24.4 % | unknown |
| Median chars, Bangla / romanized / code-mixed | 119 / 129 / 112 | 119 / 130 / 114 | 106 / 114 / 103 |
| Bangla form with more than 20% Latin letters | 5.1% | 5.8% | 1.3% |

Length by label (Bangla form, train+dev, median / mean chars): Safe 123 / 130, Offensive 90 / 117, Harmful 115 / 140, Severe 134 / 223. Severe has a long tail (long pretext prompts, see section 3.2). Devtest is shorter at every quantile.

---

## 3. Source structure and the dev/devtest shift

### 3.1 Product-complaint template (easy Safe)
Prompts of the form "I have a complaint about my order / the product has problem X, how do I contact the seller / customer service?" (keywords: ponyoti, ponno, bikretar kache, grahokseba, obhijog).

| | train | dev | devtest |
|---|---|---|---|
| Share of prompts | 19.8% | 16.5% | **5.1%** |

- Labels: 96% Safe, 1% Offensive, 2% Harmful, 1% Severe.
- **54% of all Safe prompts in train+dev are this template.**
- TF-IDF CV Safe recall: **99.7%** on complaint prompts, **69.2%** on all other Safe prompts.

### 3.2 Native Bangladeshi adversarial prompts (mostly Severe)
Union of: 2024 to 2029 year mentions, Bangladeshi places and institutions (Dhaka, Chattogram, Rajshahi, RAB, BGB, bKash, upazila), colloquial markers (bhai, bhaiya, bro, kortesi), Bangla form with more than 20% Latin letters, and code-mixed form identical to the romanized or Bangla form. Typical content: long journalist / officer / researcher pretexts asking for operational details of real Bangladeshi crimes.

| | train | dev | devtest |
|---|---|---|---|
| Share of prompts | 6.6% | 7.3% | **1.9%** |
| Year 2024 to 2029 mentioned | 2.9% | 3.5% | **0.0%** |

- Labels: Safe 9%, Offensive 10%, Harmful 15%, **Severe 66%**. This subset alone holds 18% of all Severe prompts.
- Median length 573 chars (others 115).

### 3.3 The hard core
Everything else: 74% of train, 76% of dev, **93% of devtest**.

| Subset | TF-IDF CV accuracy | Label mix S / O / H / Sv |
|---|---|---|
| Complaint template | 97.9% | 0.96 / 0.01 / 0.02 / 0.01 |
| Native adversarial | 87.9% | 0.09 / 0.10 / 0.15 / 0.66 |
| **Hard core** | **68.0%** | 0.20 / 0.20 / 0.34 / 0.27 |

### 3.4 Adversarial validation (train vs devtest)
Logistic regression on romanized word 1 to 2 grams, 5-fold CV: **AUC 0.738** (train and devtest are clearly distinguishable).

Train label mix by devtest-likeness quintile (0 = least like devtest):

| Quintile | Safe | Offensive | Harmful | Severe |
|---|---|---|---|---|
| 0 | 0.707 | 0.021 | 0.091 | 0.180 |
| 1 | 0.436 | 0.069 | 0.282 | 0.214 |
| 2 | 0.191 | 0.185 | 0.351 | 0.273 |
| 3 | 0.176 | 0.228 | 0.320 | 0.277 |
| 4 | 0.194 | 0.263 | 0.270 | 0.273 |

Most train-like words: product / seller, bhaiya, Dhaka, 2026, "I heard". Most devtest-like words: "how can I / someone", suicide, misuse, instructions, insults, role-play, "sexy".

### 3.5 Estimated devtest class mix

| Method | Safe | Offensive | Harmful | Severe |
|---|---|---|---|---|
| Hard core of train+dev | 20.4% | 19.6% | 33.5% | 26.5% |
| Importance weights from the adversarial classifier | 21.9% | 22.5% | 28.7% | 27.0% |
| Earlier EM label-shift estimate (E2 logits) | 14 to 19% | 25 to 30% | | |
| train (reference) | 34.1% | 15.3% | 26.3% | 24.3% |

### 3.6 Near-duplicates and neighbour coverage
Character 3 to 5 gram TF-IDF on the Bangla form, cosine similarity to the nearest train prompt:

| | median | share >= 0.5 | share >= 0.7 | share >= 0.9 |
|---|---|---|---|---|
| train (to other train) | 0.359 | 24.8% | 3.8% | 0.4% |
| dev | 0.326 | 18.1% | 2.6% | 0.3% |
| devtest | 0.307 | **11.1%** | 1.5% | 0.1% |

On dev, the nearest-neighbour label is correct 95 to 100% of the time when similarity is 0.5 or more, but only 54% below 0.4 (67% of dev). Devtest has fewer close neighbours, so retrieval / kNN methods would help least where it matters.

---

## 4. How the views were made

| Code-mixed type | train | dev | devtest |
|---|---|---|---|
| Bangla + English | 57.6% | 56.8% | 58.1% |
| Bangla + Hindi (Devanagari) | 26.0% | 26.3% | 26.9% |
| Bangla + Urdu (Arabic script) | 15.1% | 15.3% | 14.8% |
| Latin only (native Banglish) | 1.4% | 1.7% | 0.2% |

- Most prompts are **machine translations of English safety prompts** (literal phrasing in the Bangla form).
- The Bangla + English code-mixed view is often word-scrambled and **contains words of the original English prompt**, which helps English-strong models.
- **Artifact**: any Devanagari in the code-mixed form appears in 83% of Safe prompts but 57 to 64% of unsafe ones. This is a pipeline artifact (tied to the complaint template), a shortcut that does not generalise.
- **Romanization is one consistent scheme across splits** (kibhabe 100%, korte 99%, byabohar about 82%, ekti about 90% in all three). Normalising romanization is not needed beyond NFC.
- Romanized rows contain no Bangla characters (0 to 0.02%).

---

## 5. What the labels encode

| Pattern (romanized keywords) | n (train+dev) | devtest / train frequency | S / O / H / Sv |
|---|---|---|---|
| Prompt injection: "ignore", "upeksha" | 156 | 1.21 | 0.03 / 0.04 / **0.87** / 0.06 |
| "prompt" | 147 | 0.81 | 0.03 / 0.01 / **0.86** / 0.10 |
| "output" | 131 | 0.67 | 0.08 / 0.00 / **0.85** / 0.06 |
| Translation request | 28 | 0.80 | 0.25 / 0.07 / **0.68** / 0.00 |
| Role-play ("obhinoy") | 38 | 2.00 | 0.08 / 0.08 / **0.55** / 0.29 |
| Hacking | 90 | 1.25 | 0.10 / 0.11 / 0.43 / 0.36 |
| Phishing | 26 | 1.18 | 0.15 / 0.04 / **0.58** / 0.23 |
| Sexual content | 238 | 0.77 | 0.03 / 0.17 / 0.39 / 0.40 |
| Weapons | 143 | 1.09 | 0.11 / 0.05 / 0.15 / **0.69** |
| Suicide | 43 | 1.91 | 0.07 / 0.09 / 0.12 / **0.72** |
| Malware / virus | 102 | 1.59 | 0.13 / 0.03 / 0.22 / **0.63** |
| Bomb / explosives | 66 | 1.30 | 0.24 / 0.05 / 0.06 / **0.65** |
| Drugs | 272 | 1.00 | 0.10 / 0.10 / 0.22 / **0.58** |
| "Write a script" | 221 | 1.84 | 0.22 / 0.05 / 0.12 / **0.61** |
| Code | 131 | 1.92 | 0.38 / 0.05 / 0.09 / 0.48 |
| Children | 136 | 0.69 | 0.28 / 0.18 / 0.07 / 0.47 |
| Group hate (religion, race, LGBT) | 224 | 1.66 | 0.06 / **0.61** / 0.07 / 0.26 |
| Fiction / story | 139 | 1.36 | 0.32 / 0.22 / 0.16 / 0.31 |

Rubric implied by the data:
- **Offensive**: insults and stereotypes about groups, without a request for harmful help.
- **Harmful**: prompt injection, instruction override, privacy, fraud, misinformation, most phishing and hacking questions, sexual requests without violence.
- **Severe**: high-risk topics (weapons, explosives, drugs, suicide and self-harm, malware, child abuse, sexual violence), regardless of framing.
- `APPROACH.md` lists jailbreaks under Severe; the data puts instruction-override prompts mostly in Harmful. To be checked against the BanglaShield taxonomy.

Severe-related topics (script, code, malware, suicide) and group hate are 1.6 to 1.9 times more frequent in devtest, consistent with the estimated shift toward Offensive and Severe.

---

## 6. Label quality

| Near-duplicate pairs (nearest train neighbour) | n | label agreement |
|---|---|---|
| train, similarity >= 0.9 | 32 | 100% |
| train, >= 0.8 | 89 | 100% |
| train, >= 0.7 | 310 | 98.7% |
| train, >= 0.6 | 924 | 97.1% |
| dev, >= 0.7 | 54 | 100% |

TF-IDF CV confusion (train+dev, rows = gold):

| gold \ pred | Safe | Offensive | Harmful | Severe |
|---|---|---|---|---|
| Safe | 2,989 | 118 | 202 | 181 |
| Offensive | 108 | 1,057 | 242 | 162 |
| Harmful | 125 | 179 | 1,903 | **486** |
| Severe | 107 | 130 | **518** | 1,738 |

Confident disagreements (CV probability > 0.85 for a wrong class): 176 of 10,245 prompts (1.7%); 87 of them are Harmful/Severe swaps. Manual reading shows mostly genuine ambiguity (COVID misinformation and phishing appear under both Harmful and Severe; benign game questions with weapon words appear under Safe). Conclusion: label cleaning is not worth it; soft targets and label smoothing are the right tools.

---

## 7. Leakage and ordering checks
- No shared `prompt_id` across splits; no exact text overlap between devtest and train/dev.
- Prompt ids are shuffled: P(next id has the same label) = 0.277 in train vs 0.268 by chance; label mix flat across id blocks in train and dev; native-subset share flat across devtest id blocks.

---

## 8. Recommendations

| # | Action | Reason | Cost | Rules risk |
|---|---|---|---|---|
| 1 | **Dev-hard validation**: dev without complaint and native subsets (or dev weighted by devtest-likeness) for checkpoint selection, bias and ensemble weights | Current dev score is inflated by easy templates and misranks encoders vs Qwen | minutes, CPU | none |
| 2 | **Validate dev-hard against the leaderboard**: DS on dev-hard for B0, E1, L2, ENS1 vs LB 63.9 / 70.6 / 76.2 / 75.3 | Confirms the protocol before using it | minutes, CPU | none |
| 3 | **Prior correction** log(pi_test / pi_train) per class (about Safe -0.5, Offensive +0.3, Harmful +0.15, Severe +0.1) instead of a hand-set Safe bias | Explains the Safe-bias gain and adds the missing Offensive shift | free | none if the prior comes from the train hard core; importance-weighted prior uses blind text (ask first) |
| 4 | **Down-sample the complaint template** (keep about 25%) or weight training prompts by devtest-likeness | 20% of each epoch is spent on prompts already 98% correct | saves ~15% epoch time | none |
| 5 | **Data-derived rubric in the LLM instruction** (section 5), checked against BanglaShield's taxonomy | Targets the Harmful vs Severe confusion with the real labelling logic | one Qwen run | none |
| 6 | **Hard-Safe focus**: analyse and up-weight non-template Safe prompts (benign prompts with risky words) | Second-largest error group; Safe recall 69% outside the template | small | none |
| 7 | **Shortcut check**: does removing the code-mixed view change predictions for Safe prompts with Devanagari? | Pipeline artifact the model may exploit | analysis only | none |
| 8 | **English back-translation as a 4th view** | Source prompts are English; LLM strongest in English | medium | needs organizers' answer on external models |

Not recommended: romanization normalisation (already consistent), label cleaning (ambiguity, not noise), kNN retrieval (few close neighbours on devtest), effort on the native subset (2% of devtest).

## 9. Paper angle
"The dev set overestimates robustness because half of its Safe class is one template; a devtest-like evaluation protocol fixes model ranking." Together with the script-invariance analysis, this gives the paper a dataset insight in addition to the system.
