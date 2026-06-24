python src/offline/parse_candidates.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src/offline/feature_engineering.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src/offline/embeddings.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src/offline/build_faiss.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src/online/rank.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
