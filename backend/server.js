const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'frontend')));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'frontend', 'index.html'));
});

app.post('/predict', (req, res) => {
    const {
        income,
        credit_score,
        credit_card_usage,
        education_level,
        family_size,
        age,
        loan_amount,
    } = req.body;

    const pythonPath = path.join(__dirname, 'ml/predict.py');

    const pythonProcess = spawn('python', [pythonPath]);

    const inputData = JSON.stringify({
        income: Number(income),
        credit_score: Number(credit_score),
        credit_card_usage: Number(credit_card_usage),
        education_level: Number(education_level),
        family_size: Number(family_size),
        age: Number(age),
        loan_amount: Number(loan_amount),
    });

    pythonProcess.stdin.write(inputData);
    pythonProcess.stdin.end();

    let stdout = '';
    let stderr = '';

    pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString();
        console.error('Python stderr:', data.toString());
    });

    pythonProcess.on('close', (code) => {
        if (code !== 0) {
            return res.status(500).json({
                error: 'Python process failed',
                details: stderr || null,
            });
        }

        try {
            const result = JSON.parse(stdout);
            res.json(result);
        } catch (err) {
            res.status(500).json({
                error: 'Invalid response from model',
                raw: stdout,
            });
        }
    });
});

app.listen(PORT, () => {
    console.log(`Server running on ${PORT}`);
});