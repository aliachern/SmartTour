document.getElementById('preferencesForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const tripType = e.target.tripType.value;
    const budget = e.target.budget.value;

    fetch('https://your-ml-api-url.com/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tripType, budget })
    })
    .then(res => res.json())
    .then(data => {
        const output = document.getElementById('recommendations');
        output.innerHTML = '<h2>Recommended Destinations:</h2>' + 
            data.recommendations.map(r => '<p>' + r + '</p>').join('');
    });
});
