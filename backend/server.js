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

    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';

    const payload = {
        income: Number(income),
        credit_score: Number(credit_score),
        credit_card_usage: Number(credit_card_usage),
        education_level: Number(education_level),
        family_size: Number(family_size),
        age: Number(age),
        loan_amount: Number(loan_amount),
    };

    // validate so NaN doesn't get passed to ML
    for (const [k, v] of Object.entries(payload)) {
        if (!Number.isFinite(v)) {
            return res.status(400).json({ error: `Invalid number for ${k}`, received: req.body });
        }
    }

    let responded = false;
    const respond = (status, body) => {
        if (responded) return;
        responded = true;
        res.status(status).json(body);  
    };

    const pythonProcess = spawn(pythonCmd, [pythonPath], {
        cwd: path.join(__dirname, 'ml'), 
    });

    pythonProcess.on('error', (err) => {
        respond(500, {
            error: 'Failed to start Python',
            details: err.message,
            pythonCmd,
            pythonPath,
        });
    });

    // Timeout guard (prevents Railway 502 from hanging)
    const t = setTimeout(() => {
        try { pythonProcess.kill('SIGKILL'); } catch {}
        respond(500, { error: 'Python timed out' });
    }, 8000);

    pythonProcess.stdin.write(JSON.stringify(payload));
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
        clearTimeout(t);

        if (code !== 0) {
            return respond(500, {
                error: 'Python process failed',
                details: stderr || null,
            });
        }

        try {
            const result = JSON.parse(stdout);
            respond(200, result);
        } catch (err) {
            respond(500, {
                error: 'Invalid response from model',
                raw: stdout,
                details: stderr || null,
            });
        }
    });
});


app.listen(PORT, () => {
    console.log(`Server running on ${PORT}`);
});