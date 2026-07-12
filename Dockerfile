FROM python:3.8-slim

WORKDIR /research

COPY . .

LABEL org.opencontainers.image.title="Adjustable Context-Aware Transformer"
LABEL org.opencontainers.image.description="Research code and reproducibility materials for adjustable context-aware attention in multi-horizon time-series forecasting."
LABEL org.opencontainers.image.source="https://github.com/SepKfr/Adjustable-context-aware-transfomer"
LABEL org.opencontainers.image.documentation="https://github.com/SepKfr/Adjustable-context-aware-transfomer#readme"
LABEL org.opencontainers.image.licenses="MIT"

CMD ["sh", "-c", "echo 'Adjustable Context-Aware Transformer research artifact'; echo 'See README.md for dependencies, datasets, and experiment instructions.'; ls -la"]
