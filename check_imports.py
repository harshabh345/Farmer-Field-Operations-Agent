import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pkgs = [
    ('chromadb',             'chromadb'),
    ('rank_bm25',            'rank_bm25'),
    ('sentence_transformers','sentence_transformers'),
    ('numpy',                'numpy'),
    ('pandas',               'pandas'),
    ('joblib',               'joblib'),
    ('xgboost',              'xgboost'),
    ('requests',             'requests'),
    ('dotenv',               'dotenv'),
    ('groq',                 'groq'),
    ('langgraph',            'langgraph'),
    ('streamlit',            'streamlit'),
    ('folium',               'folium'),
    ('streamlit_folium',     'streamlit_folium'),
]

all_ok = True
for name, mod in pkgs:
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', 'ok')
        print(f'OK       {name:<28} {ver}')
    except ImportError as e:
        print(f'MISSING  {name:<28} {e}')
        all_ok = False

print()
print('ALL PACKAGES OK' if all_ok else 'SOME PACKAGES MISSING')
