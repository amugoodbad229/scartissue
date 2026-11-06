# syntax=docker/dockerfile:1
# Static workbench: no model, no GPU, no key, no network calls at runtime.
FROM python:3.12-slim

WORKDIR /app
COPY space-static/ ./space-static/

RUN useradd --create-home --uid 10001 judge
USER judge

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/index.html', timeout=2)"

CMD ["python", "-m", "http.server", "8000", "--bind", "0.0.0.0", "--directory", "space-static"]
