# Potato Disease Classification

A high-accuracy deep learning application for classifying potato leaf diseases using PyTorch & Transfer Learning (ResNet18). The project features a Flask REST API for model serving and a responsive React frontend for image upload and results display.



## Table of Contents

- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

---

## Key Features

- **High-Accuracy Classification:** Utilizes a ResNet18 model fine-tuned for this specific task, achieving over 99.5% validation accuracy.
- **Confidence Score:** Provides a confidence level for each prediction, giving users more insight into the model's certainty.
- **RESTful API:** The backend is built with Flask, providing a clean and robust API for model inference.
- **Interactive & Responsive UI:** A modern frontend built with React and Vite allows for easy drag-and-drop file uploads and dynamic results display.
- **Comparative Model Analysis:** The final model was chosen after a rigorous comparison against custom CNN architectures, demonstrating a data-driven approach to model selection.

---

## Tech Stack

- **Backend:** Flask, PyTorch
- **Frontend:** React, Vite, JavaScript, HTML, CSS
- **Machine Learning:** PyTorch, ResNet18 (Transfer Learning)

---

## Model Performance

- **Architecture:** ResNet18, pre-trained on ImageNet, fine-tuned on potato disease dataset.
- **Final Validation Accuracy:** **99.53%**
- **Classes:**
  - `Potato___Early_blight`
  - `Potato___Late_blight`
  - `Potato___healthy`

---

## Project Structure

```
TuberMD/
├── backend/
│   ├── app.py              # Flask server entrypoint
│   ├── utils.py            # PyTorch model loading and prediction logic
│   ├── requirements.txt    # Python dependencies
│   ├── uploads/            # Temporary storage for uploaded images
│   └── models/
│       └── best_potato_disease_model.pth # The trained PyTorch model
│
└── frontend/
    ├── src/
    │   ├── App.jsx         # Main React component
    │   ├── App.css         # Styles for the App component
    │   ├── main.jsx        # React entrypoint
    │   └── index.css       # Global styles
    ├── index.html          # Vite entrypoint HTML
    ├── package.json        # Node.js dependencies
    └── vite.config.js      # Vite configuration
```

---

## Setup and Installation

### Prerequisites

- Python 3.1+
- Node.js and npm (or yarn)
- A trained model file: `best_potato_disease_model.pth`

---

### 1. Backend Setup

```bash
# Clone the repository
git clone https://github.com/your-username/TuberMD.git
cd TuberMD/backend

# Create and activate a virtual environment
python -m venv venv
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# IMPORTANT: Place your trained model file in this directory:
# backend/models/best_potato_disease_model.pth

# Run the Flask server
python app.py
```

The backend will now be running on [http://localhost:5000](http://localhost:5000).

---

### 2. Frontend Setup

Open a new terminal for the frontend.

```bash
# Navigate to the frontend directory
cd TuberMD/frontend

# Install the required Node.js packages
npm install

# Start the Vite development server
npm run dev
```

The frontend will now be running on [http://localhost:5173](http://localhost:5173) (or the next available port).

---

## Usage

1. Ensure both the backend and frontend servers are running.
2. Open your web browser and navigate to [http://localhost:5173](http://localhost:5173).
3. Drag and drop an image of a potato leaf onto the designated area, or click to open the file selector.
4. Click the **"Classify Leaf"** button.
5. The predicted disease and the model's confidence level will be displayed on the screen.

---

## Future Improvements

- [ ] Deploy the application to a cloud service like Heroku or AWS.
- [ ] Expand the model to classify diseases for other plants (e.g., tomato, bell pepper).
- [ ] Implement batch processing to allow users to upload multiple images at once.
- [ ] Develop a mobile-first version of the application using React Native.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---


