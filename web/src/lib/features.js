// How each model input is shown in the form.
// The numbers themselves (ranges, averages, coefficients) come from
// src/data/model.json, which is written by `python compare_models.py`.

export const GROUPS = [
  {
    id: 'patient',
    title: 'About the patient',
    hint: 'Basic measurements taken at the first visit.',
    fields: ['age', 'sex', 'bmi', 'bp'],
  },
  {
    id: 'blood',
    title: 'Blood test results',
    hint: 'Six blood serum measurements from the same visit.',
    fields: ['s1', 's2', 's3', 's4', 's5', 's6'],
  },
]

export const FIELD_INFO = {
  age: { label: 'Age', unit: 'years', step: 1 },
  sex: {
    label: 'Sex',
    unit: '',
    options: [
      { value: 1, label: 'Male' },
      { value: 2, label: 'Female' },
    ],
    // The dataset only stores 1 or 2. This mapping is the usual assumption.
    note: 'Assumed coding: 1 = male, 2 = female. The dataset does not document it.',
  },
  bmi: { label: 'Body mass index', unit: 'kg/m²', step: 0.1 },
  bp: { label: 'Average blood pressure', unit: 'mm Hg', step: 1 },
  s1: { label: 'Total cholesterol', unit: 'mg/dL', step: 1 },
  s2: { label: 'LDL cholesterol', unit: 'mg/dL', step: 1 },
  s3: { label: 'HDL cholesterol', unit: 'mg/dL', step: 1 },
  s4: { label: 'Total cholesterol ÷ HDL', unit: 'ratio', step: 0.1 },
  s5: { label: 'Triglycerides (log)', unit: 'log scale', step: 0.05 },
  s6: { label: 'Blood sugar', unit: 'mg/dL', step: 1 },
}

// Show at most 2 decimals and drop trailing zeros: 84.33 -> "84.33", 101 -> "101"
export function formatValue(value) {
  return String(Number(value.toFixed(2)))
}
