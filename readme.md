# Fingerprint Detection

Fingerprint Detection is a Python project for education use only.

## Installation

Download the dataset [here](https://www.kaggle.com/datasets/ruizgara/socofing).

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The project exposes a FastAPI server with endpoints to register and identify fingerprints.  A helper CLI is also provided for registration.

### Register a fingerprint from the command line

```bash
python -m src.fingerprint.presentation.cli.register_person \
    --image path/to/image.png \
    --person-id P001 \
    --name "Alice" \
    --document "12345678"
```

Minutiae can also be provided in JSON form with `--minutiae`.

### Run the API with Docker

Use the provided `docker-compose.yml` to launch the API together with PostgreSQL and the FAISS index volume:

```bash
docker compose up --build
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
