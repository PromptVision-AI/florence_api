import uvicorn
from app.api.routes import app
from app.core.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "app.api.routes:app",
        host=API_HOST,
        port=API_PORT,
        reload=False
    ) 