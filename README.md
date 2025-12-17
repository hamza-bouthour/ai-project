# Car Loan Approval Assistant

A web application that predicts car loan approval based on user financial and demographic information, combined with automated car value estimation.

## Features

- User inputs financial info and selects a car
- Automatic car value estimation (mocked)
- Machine learning prediction for loan approval
- Simple web interface

## Technology Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Node.js, Express
- **ML**: Python, Scikit-learn, Pandas
- **Communication**: JSON over HTTP

## Setup

1. Ensure Node.js and Python are installed.

2. Install Python dependencies:
   ```
   pip install pandas scikit-learn joblib
   ```

3. Install Node.js dependencies:
   ```
   cd backend
   npm install
   ```

4. Train the model:
   ```
   cd ml
   python train.py
   ```

5. Start the server:
   ```
   cd backend
   npm start
   ```

6. Open browser to http://localhost:3000

## Project Structure

- `backend/`: Node.js server
- `frontend/`: HTML/CSS/JS client
- `ml/`: Python ML scripts
- `data/`: Dataset