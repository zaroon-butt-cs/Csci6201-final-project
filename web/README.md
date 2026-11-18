# Diabetes Progression Predictor (web app)

A Vite + React app that estimates one-year diabetes progression from ten
baseline measurements. It uses the model selected by `compare_models.py`
in the project root.

## Run

```bash
cd web
npm install
npm run dev        # open the address it prints (http://localhost:5173)
```

`npm run build` creates the production version in `web/dist/`.

## How the prediction works

The selected model is a scaled linear model, which is just a formula, so the
app calculates it directly in the browser. No server is needed.

```
z_i        = (value_i - mean_i) / scale_i
prediction = intercept + sum(coefficient_i * z_i)
```

All numbers (means, scales, coefficients, input ranges, example patients)
are in `src/data/model.json`. That file is written by
`python compare_models.py`, so re-running the comparison updates the app.

## Files

```
src/
├── App.jsx                  # page layout and state
├── index.css                # all styling (light and dark theme, responsive)
├── data/model.json          # exported model (generated, do not edit)
├── lib/
│   ├── predict.js           # the prediction formula
│   ├── features.js          # labels, units and step size of each input
│   └── useAnimatedNumber.js # smooth number animation
└── components/
    ├── NumberField.jsx      # spinner (− value +) with a slider
    ├── ChoiceField.jsx      # two-option switch
    ├── Gauge.jsx            # half-circle gauge
    └── Drivers.jsx          # which inputs raise or lower the prediction
```
