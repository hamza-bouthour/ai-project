const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../frontend')));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

app.post('/predict', (req, res) => {
    const { income, credit_score, credit_card_usage, education_level, family_size, age, loan_amount } = req.body;

    // Call Python script
    const pythonProcess = spawn('python', [path.join(__dirname, '../ml/predict.py')], {
        stdio: ['pipe', 'pipe', 'pipe']
    });

    const inputData = JSON.stringify({
        income: parseFloat(income),
        credit_score: parseFloat(credit_score),
        credit_card_usage: parseFloat(credit_card_usage),
        education_level: parseInt(education_level),
        family_size: parseInt(family_size),
        age: parseFloat(age),
        loan_amount: parseFloat(loan_amount)
    });

    pythonProcess.stdin.write(inputData);
    pythonProcess.stdin.end();

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`Python error: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        if (code === 0) {
            try {
                const result = JSON.parse(output);
                res.json(result);
            } catch (e) {
                res.status(500).json({ error: 'Failed to parse prediction result' });
            }
        } else {
            res.status(500).json({ error: 'Prediction failed' });
        }
    });
});

app.listen(PORT, () => {
    console.log(`Server running on ${PORT}`);
});