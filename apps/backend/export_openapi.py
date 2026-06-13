import json
from src.main import app

def export_openapi():
    openapi_data = app.openapi()
    with open("openapi.json", "w") as f:
        json.dump(openapi_data, f, indent=2)
    print("OpenAPI spec exported to openapi.json")

if __name__ == "__main__":
    export_openapi()
