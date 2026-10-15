import warnings, os, sys
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

print('=== SYSTEM VALIDATION ===')

# 1. Packages
try:
    import fastapi, uvicorn, pydantic, joblib, numpy, sklearn, xgboost, lightgbm
    print('[OK] All packages importable: fastapi uvicorn pydantic joblib sklearn xgboost lightgbm')
except Exception as e:
    print('[FAIL] Import:', e)

# 2. Model
try:
    import joblib, numpy as np, json
    m = joblib.load('backend/models/model.pkl')
    s = joblib.load('backend/models/scaler.pkl')
    meta = json.load(open('backend/models/model_metadata.json'))
    print('[OK] Model v%s | acc=%.2f%% | AUC=%.4f | samples=%d | models=%s' % (
        meta['version'], meta['accuracy'], meta['auc_roc_macro'],
        meta['training_samples'], meta.get('base_models',[])))
except Exception as e:
    print('[FAIL] Model:', e)

# 3. Predictions
try:
    FC=['fatigue','excessive_thirst','blurred_vision','frequent_urination','numbness','weight_loss','slow_healing','skin_darkening','headache','chest_pain','shortness_breath','palpitations','dizziness','ankle_swelling','anxiety','sleep_issues','low_mood','irritability','weight_gain','cold_intolerance','hair_loss','nausea','irregular_periods','acne','glucose','blood_pressure','age','bmi','sleep_hrs','hba1c','family_diabetes','family_bp','smoking','sedentary']
    IDX={c:i for i,c in enumerate(FC)}
    CN=['Normal','Diabetes','Hypertension','MentalHealth','Thyroid','PCOS']
    def eng(X):
        g=(X[:,IDX['glucose']]-70)/250;b=(X[:,IDX['blood_pressure']]-80)/130
        an=(X[:,IDX['age']]-14)/66;bn=(X[:,IDX['bmi']]-16)/32
        sl=(X[:,IDX['sleep_hrs']]-2)/8;h=(X[:,IDX['hba1c']]-4.5)/8.5
        ms=(X[:,IDX['fatigue']]+X[:,IDX['excessive_thirst']]+X[:,IDX['frequent_urination']]+X[:,IDX['numbness']]+X[:,IDX['slow_healing']]+g+h)/7
        cs=(X[:,IDX['headache']]+X[:,IDX['chest_pain']]+X[:,IDX['shortness_breath']]+X[:,IDX['palpitations']]+X[:,IDX['dizziness']]+b+an)/7
        mts=(X[:,IDX['anxiety']]+X[:,IDX['sleep_issues']]+X[:,IDX['low_mood']]+X[:,IDX['irritability']])/4
        hs=(X[:,IDX['weight_gain']]+X[:,IDX['cold_intolerance']]+X[:,IDX['hair_loss']]+X[:,IDX['irregular_periods']]+X[:,IDX['acne']])/5
        return np.hstack([X,np.column_stack([ms,cs,mts,hs,np.clip(g*bn,0,1),np.clip(b*an,0,1),np.clip((1-sl)*X[:,IDX['anxiety']],0,1),np.clip(h*g,0,1),np.mean(X[:,:24],axis=1),(X[:,IDX['excessive_thirst']]+X[:,IDX['frequent_urination']]+X[:,IDX['blurred_vision']])/3,(X[:,IDX['cold_intolerance']]+X[:,IDX['hair_loss']]+X[:,IDX['weight_gain']])/3,(X[:,IDX['irregular_periods']]+X[:,IDX['acne']]+X[:,IDX['weight_gain']])/3])])
    def pred(syms, g=90, bp=110, age=35, bmi=24, sl=7, h=5.0, gender='male'):
        sv=set(syms); v=[1.0 if x in sv else 0.0 for x in FC[:24]]
        if gender=='male': v[22]=0.0; v[23]=0.0
        v+=[g,bp,age,bmi,sl,h,0,0,0,0]
        X=np.array([v],dtype=np.float32)
        p=m.predict_proba(s.transform(eng(X)))[0]
        return CN[np.argmax(p)], int(round(max(p)*100)), int(round(p[5]*100))

    tests = [
        ('Diabetes (male)',         ['excessive_thirst','frequent_urination','blurred_vision'], dict(g=180,h=8.0,age=45,gender='male'),   'Diabetes'),
        ('Hypertension (male)',     ['headache','chest_pain','shortness_breath'],               dict(bp=165,age=50,gender='male'),          'Hypertension'),
        ('Mental (female)',         ['anxiety','sleep_issues','low_mood','irritability'],       dict(age=25,gender='female'),               'MentalHealth'),
        ('Thyroid (female)',        ['fatigue','cold_intolerance','hair_loss','weight_gain'],   dict(age=35,gender='female'),               'Thyroid'),
        ('PCOS (female)',           ['irregular_periods','acne','weight_gain'],                 dict(age=22,gender='female'),               'PCOS'),
        ('Normal (male)',           [],                                                         dict(g=85,bp=110,age=28,bmi=23,gender='male'),'Normal'),
        ('Mental no PCOS (female)', ['anxiety','sleep_issues','low_mood'],                     dict(age=28,gender='female'),               'MentalHealth'),
        ('Chest pain (male)',       ['chest_pain'],                                             dict(age=35,gender='male'),                 'Normal'),
    ]
    passed=0
    for name,syms,kw,expected in tests:
        top,score,pcos = pred(syms,**kw)
        ok = top==expected
        if ok: passed+=1
        print('  [%s] %-28s -> %-14s %3d%% | PCOS=%d%%%s' % ('OK' if ok else 'FAIL', name, top, score, pcos, '' if ok else '  EXPECTED:'+expected))
    print('[%s] Predictions: %d/%d passed' % ('OK' if passed==len(tests) else 'FAIL', passed, len(tests)))
except Exception as e:
    print('[FAIL] Prediction test:', e)
    import traceback; traceback.print_exc()

# 4. FastAPI app + routes
try:
    from backend.main import app, _model, _scaler, _model_version
    routes = [r.path for r in app.routes]
    needed = ['/', '/health', '/api/predict', '/api/agent/suggest', '/api/model-info', '/api/patients', '/api/medication/log']
    missing = [r for r in needed if r not in routes]
    print('[OK] FastAPI app | model=%s | version=%s' % ('LOADED' if _model else 'FALLBACK', _model_version[:40]))
    if missing: print('[WARN] Missing routes:', missing)
    else: print('[OK] All %d routes present: %s' % (len(needed), needed))
except Exception as e:
    print('[FAIL] FastAPI:', e)

print('=== VALIDATION COMPLETE ===')
