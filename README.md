# Florence API Service

A FastAPI-based service that leverages the Florence-2 model for advanced image and text processing. This project provides a containerized API service that can be easily deployed using Docker.

## 🚀 Features

- FastAPI-based REST API
- Florence-2 model integration
- GPU acceleration support
- Containerized deployment
- Environment variable configuration

## 📋 Prerequisites

- Docker
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed

## 🛠️ Project Structure

```
florence_api/
├── app/                    # Application code
├── model/                  # Florence model files
├── docker/                 # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── main.py                # FastAPI application entry point
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables
```

## 🔧 Configuration

Create a `.env` file in the root directory with the following variables:
```env
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=/app/model
```

## 🚀 Running with Docker

1. **Build the Docker image**
   ```bash
   cd docker
   docker compose build --no-cache
   ```

2. **Start the service**
   ```bash
   docker compose up
   ```

   The API will be available at `http://localhost:8000`

3. **Stop the service**
   ```bash
   docker compose down
   ```

## 🔍 API Documentation

Once the service is running, you can access:
- API documentation: `http://localhost:8000/docs`
- Alternative documentation: `http://localhost:8000/redoc`

## 🛟 Troubleshooting

### CUDA Version Issues
Make sure your NVIDIA driver supports the CUDA version specified in the Dockerfile. You can check your supported CUDA version with:
```bash
nvidia-smi
```

### GPU Access Issues
Ensure the NVIDIA Container Toolkit is properly installed:
```bash
nvidia-container-cli info
```

## 📝 License

[Your License Here]

## 🤝 Contributing

[Your Contributing Guidelines Here] 