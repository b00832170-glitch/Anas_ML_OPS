\# Anas\_ML\_OPS — Credit Risk Prediction + LLM Assistant



A machine learning model that predicts credit default risk, connected to a 

Groq LLM (Llama 3.3) that can answer questions in natural language.



\## Architecture

Dataset CSV → Training (scikit-learn) → model.pkl

↓

predict.py (ML tool)

↓

Groq LLM (Llama 3.3-70b)

↓

chat.py (natural language)

↓

User



## \## Installation

## 

## \### 1. Clone the repository

## ```bash

## git clone https://github.com/b00832170-glitch/Anas\_ML\_OPS.git

## cd Anas\_ML\_OPS

## ```

## 

## \### 2. Install dependencies

## ```bash

## pip install -r requirements.txt

## ```

## 

## \### 3. Train the model

## ```bash

## python ml/train.py

## ```

## 

## \### 4. Set up your Groq API key

## \- Create a free account at https://console.groq.com

## \- Generate an API key

## \- Create a `.env` file in the project root:



GROQ\_API\_KEY=your\_groq\_api\_key\_here



\## 5. Start the AI Assistant

```bash

python chat.py

```



\## Example Questions



Once the assistant is running, you can ask:



Is a 25-year-old renter with $50k income and a $10k personal loan grade B risky?

Analyze this borrower: 22 years old, $9k income, grade F loan of $35k, past default on file

What is the model accuracy?



\## Dataset



\- \*\*Source\*\* : `data/credit\_risk\_dataset.csv`

\- \*\*Target\*\* : `loan\_status` (0 = no default, 1 = default)

\- \*\*Model\*\* : RandomForestClassifier

\- \*\*Size\*\* : 32,581 rows



\## Model Performance



| Metric | Score |

|---|---|

| Accuracy | 92.96% |

| ROC-AUC | 92.46% |



\## Features



| Feature | Type | Description |

|---|---|---|

| person\_age | int | Borrower age |

| person\_income | float | Annual income |

| person\_home\_ownership | str | RENT / OWN / MORTGAGE / OTHER |

| person\_emp\_length | float | Employment length (years) |

| loan\_intent | str | Loan purpose |

| loan\_grade | str | Loan grade (A-G) |

| loan\_amnt | float | Loan amount |

| loan\_int\_rate | float | Interest rate (%) |

| loan\_percent\_income | float | Loan/income ratio |

| cb\_person\_default\_on\_file | str | Historical default (Y/N) |

| cb\_person\_cred\_hist\_length | int | Credit history length |



\## Project Structure

