document.addEventListener('DOMContentLoaded', () => {
  const form = document.querySelector('#prediction-form');
  if (!form) return;
  form.addEventListener('submit', async (event) => {
    event.preventDefault(); const button = form.querySelector('button'); button.disabled = true; button.textContent = 'Assessing…';
    try { const payload = Object.fromEntries(new FormData(form)); const response = await fetch('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)}); const data = await response.json(); const result = document.querySelector('#prediction-result'); result.classList.remove('hidden'); result.innerHTML = `<h3>${data.risk_level} risk · ${data.confidence}% confidence</h3><p>${data.recommendation}</p>`; result.scrollIntoView({behavior:'smooth',block:'nearest'}); } catch (_) { alert('Prediction could not be completed. Please check the entered values.'); } finally { button.disabled = false; button.innerHTML = 'Assess risk <span>→</span>'; }
  });
});
