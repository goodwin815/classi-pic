# ClassiPic

**ClassiPic** is an advanced image classification application that leverages a pre-trained deep learning model to identify and categorize images. This project integrates a React frontend with a Flask backend to deliver a seamless and efficient user experience for real-time image classification.

## Features

- **User-Friendly Interface:** Built with React, providing a responsive design for easy image uploads.
- **Advanced Backend:** Utilizes Flask to integrate a pre-trained TensorFlow model for accurate predictions.
- **Real-Time Predictions:** Offers quick feedback on image classifications.
- **Scalable and Secure:** Containerized using Docker for easy deployment and scalability.
- **Robust Error Handling:** Manages invalid inputs and server errors gracefully.

## Technology Stack

- **Frontend:** React, Axios
- **Backend:** Flask, TensorFlow, Flask-CORS
- **Deployment:** Docker, Docker-Compose

## Getting Started

### Prerequisites

- Next.js 14.x
- React
- Flask
- Python 3.x
- Docker
- Git

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/classipic.git
   cd classipic
   ```

2. **Set up the backend:**

    ```sh
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Set up the frontend:**

    ```sh
    cd ../frontend
    npm install
    ```

## Running Locally

1. **Start the Flask backend:**

    ```sh
    cd backend
    flask run
    ```

2. **Start the Next.js frontend:**

    ```sh
    cd frontend
    yarn start
    ```

## Docker Deployment

1. **Build and run the Docker containers:**

    ```sh
    docker-compose up --build
    ```

## Usage

- Navigate to `http://localhost:3000` in your web browser.
- Upload an image to get real-time predictions.

## Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Flask Documentation](https://flask.palletsprojects.com/en/3.0.x/)
- [Tensorflow Hub](https://tfhub.dev/)
- [Docker Documentation](https://docs.docker.com/)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE]() file for details.
