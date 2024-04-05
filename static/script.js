function displayPentagonChart(predictions) {
    const pentagonLabels = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness'];
    const pentagonData = predictions[0]; // Use the first set of predictions for the chart
    
    const ctx = document.getElementById('pentagon-chart').getContext('2d');
    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: pentagonLabels,
            datasets: [{
                label: 'Personality Traits',
                data: pentagonData,
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                pointBackgroundColor: 'rgba(75, 192, 192, 1)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(75, 192, 192, 1)'
            }]
        },
        options: {
            scale: {
                ticks: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });

    // Display the chart container
    const chartContainer = document.getElementById('pentagon-chart-container');
    chartContainer.style.display = 'block';
}

function displaySentimentChart(sentimentScores) {
    const ctx = document.getElementById('sentiment-chart').getContext('2d');
    
    // Determine the color based on sentiment score
    const color = getColorForSentiment(sentimentScores);

    // Create the bar chart data
    const data = {
        labels: ['Sentiment Score'],
        datasets: [
            {
                data: [sentimentScores],
                backgroundColor: color, // Use the determined color
                borderColor: color,
                borderWidth: 1,
            },
        ],
    };

    new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        stepSize: 0.2,
                    },
                },
            },
            plugins: {
                legend: {
                    display: false,
                },
            },
        },
    });
    
    // Display the sentiment score
    const sentimentScoreElement = document.getElementById('sentiment-score');
    sentimentScoreElement.textContent = sentimentScores.toFixed(2);
}

function getColorForSentiment(sentimentScores) {
    // Determine the color based on the sentiment score
    if (sentimentScores > 0.2) {
        return 'rgba(75, 192, 75, 0.7)'; // Green for positive sentiment
    } else if (sentimentScores < -0.2) {
        return 'rgba(255, 99, 132, 0.7)'; // Red for negative sentiment
    } else {
        return 'rgba(192, 192, 192, 0.7)'; // Gray for neutral sentiment
    }
}

// Function to get sentiment label
function getSentimentLabel(sentimentScore) {
    if (sentimentScore > 0.2) {
        return "Positive";
    } else if (sentimentScore < -0.2) {
        return "Negative";
    } else {
        return "Neutral";
    }
}

function predict() {
    const inputText = document.getElementById('input-text').value;
    const inputData = inputText.split('\n').filter(sentence => sentence.trim() !== '');
    
    // Send data to the backend for prediction
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ data: inputData })
    })
    .then(response => response.json())
    .then(response => {
        const predictions = response.predictions;
        const sentimentScores = response.sentiment_scores;
        const sentimentLabel = getSentimentLabel(sentimentScores);
        // Display the predicted personality trait scores
        const outputDiv = document.getElementById('output');
        outputDiv.innerHTML = '<h2>Predicted Scores:</h2>';
        for (let i = 0; i < inputData.length; i++) {
            outputDiv.innerHTML += `
                <div class="col-md-6">
                    <div class="result-card">
                        <h3>Big-five personality trait scores:</h3>
                        <ul>
                            <li>Extraversion: ${predictions[i][0].toFixed(2)}</li>
                            <li>Neuroticism: ${predictions[i][1].toFixed(2)}</li>
                            <li>Agreeableness: ${predictions[i][2].toFixed(2)}</li>
                            <li>Conscientiousness: ${predictions[i][3].toFixed(2)}</li>
                            <li>Openness: ${predictions[i][4].toFixed(2)}</li>
                        </ul>
                        <!-- Display sentiment score -->
                        <h3>Sentiment score:</h3>
                        <p>Sentiment Score: <span>${sentimentScores[i].toFixed(2)} (${sentimentLabel})</span></p>
                        <!-- Update the progress bar for sentiment score -->
                        <div class="progress">
                            <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: ${(sentimentScores[i] + 1) * 50}%;"></div>
                        </div>
                    </div>
                </div>`;
        }
        outputDiv.style.display = 'block';
        displayPentagonChart(predictions);
        displaySentimentChart(sentimentScores);
    })
    .catch(error => console.error('Error:', error));
}