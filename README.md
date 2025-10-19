### Quality Metrics
This project adds automatic dataset health checks:

| Metric | Description | Why it Matters |
|:--|:--|:--|
| Toxicity Score | Probability of harmful language via Detoxify | Filters unsafe content before training |
| Factual Markers | Heuristic check for informational content | Prioritizes fact-based text |
| Domain Coverage | Topic distribution across news domains | Ensures dataset diversity |

> These metrics mirror what a Foundation Data TPM would monitor in production to evaluate pre-training corpora quality.

After cleaning and filtering, the dataset retains 4,141 high-quality English samples. Toxicity levels are negligible, factual content makes up roughly a third of the corpus, and the domain distribution indicates a technology-heavy skew.

App - https://divijmathur-foundation-data-demo-app-oq5hoa.streamlit.app/