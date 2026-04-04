from api.server import app
import uvicorn


def main() -> None:
    """Entry point for project script checks."""
    uvicorn.run("api.server:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()