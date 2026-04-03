\# Anas\_ML\_OPS — Credit Risk Prediction + LLM Assistant



A machine learning model that predicts credit default risk, connected to a Groq LLM (Llama 3.3) that answers questions in natural language.



\## Architecture



&#x20;   Dataset CSV → Training (scikit-learn) → model.pkl

&#x20;                                                ↓

&#x20;                                         predict.py (ML tool)

&#x20;                                                ↓

&#x20;                                      Groq LLM (Llama 3.3-70b)

&#x20;                                                ↓

&#x20;                                     chat.py (natural language)

&#x20;                                                ↓

&#x20;                                             User



\## Installation



\*\*1. Clone the repository\*\*



&#x20;   git clone https://github.com/b00832170-glitch/Anas\_ML\_OPS.git

&#x20;   cd Anas\_ML\_OPS



\*\*2. Install dependencies\*\*



&#x20;   pip install -r requirements.txt



\*\*3. Train the model\*\*



&#x20;   python ml/train.py



\*\*4. Set up your Groq API key\*\*



Create a free account at https://console.groq.com, generate an API key, then create a `.env` file in the project root:



&#x20;   GROQ\_API\_KEY=your\_groq\_api\_key\_here



\*\*5. Start the AI Assistant\*\*



&#x20;   python chat.py



\## Example Questions



&#x20;   Is a 25-year-old renter with $50k income and a $10k personal loan grade B risky?

&#x20;   Analyze this borrower: 22 years old, $9k income, grade F loan of $35k, past default on file

&#x20;   What is the model accuracy?



\## Model Performance



| Metric | Score |

|---|---|

| Accuracy | 92.96% |

| ROC-AUC | 92.46% |



\## Dataset



\- \*\*Source\*\* : data/credit\_risk\_dataset.csv

\- \*\*Target\*\* : loan\_status (0 = no default, 1 = default)

\- \*\*Model\*\* : RandomForestClassifier

\- \*\*Size\*\* : 32,581 rows



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



&#x20;   Anas\_ML\_OPS/

&#x20;   ├── data/

&#x20;   │   └── credit\_risk\_dataset.csv

&#x20;   ├── ml/

&#x20;   │   ├── train.py

&#x20;   │   └── predict.py

&#x20;   ├── server/

&#x20;   │   ├── main.py

&#x20;   │   └── mcp\_server.py

&#x20;   ├── chat.py

&#x20;   ├── .env.example

&#x20;   ├── requirements.txt

&#x20;   └── README.md

